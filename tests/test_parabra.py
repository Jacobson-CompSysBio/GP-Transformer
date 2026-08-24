import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.parabra import (
    aggregate_state_predictions,
    author_gaussian_kernel,
    environment_scores,
    fixed_equal_zscore_ensemble,
    fit_full_rank_megasem,
    fit_variance_ridge_dual,
    fit_variance_ridge_primal,
    kernel_feature_gram,
    mean_impute_markers,
    within_environment_zscore,
)
from scripts.train_parabra import load_inputs, save_scores


class ParabraKernelTests(unittest.TestCase):
    def test_mean_imputation_uses_each_marker_mean(self):
        markers = np.array([[0.0, np.nan], [1.0, 0.5], [0.5, 1.0]])
        actual, count = mean_impute_markers(markers)

        self.assertEqual(count, 1)
        self.assertAlmostEqual(float(actual[0, 1]), 0.75)
        self.assertTrue(np.isfinite(actual).all())

    def test_author_kernel_matches_direct_euclidean_formula(self):
        markers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        actual, mean_distance = author_gaussian_kernel(markers, phi=1.0)

        distances = np.linalg.norm(markers[:, None, :] - markers[None, :, :], axis=2)
        expected_mean = distances.sum() / (len(markers) * (len(markers) - 1))
        expected = np.exp(-distances / expected_mean)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(mean_distance, float(expected_mean), places=6)

    def test_k2x_feature_gram_equals_kernel_square(self):
        kernel = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
        actual = kernel_feature_gram(kernel)
        np.testing.assert_allclose(actual, kernel @ kernel.T, rtol=0, atol=1e-7)


class ParabraInputTests(unittest.TestCase):
    def test_mixed_missing_plot_cell_stays_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Training_data").mkdir()
            (root / "Testing_data").mkdir()
            traits = pd.DataFrame(
                {
                    "Env": ["AA_2014"] * 4 + ["AA_2023"] * 2,
                    "Year": [2014] * 4 + [2023] * 2,
                    "Hybrid": ["H1", "H1", "H2", "H3", "H1", "H2"],
                    "Yield_Mg_ha": [1.0, np.nan, 2.0, 4.0, 5.0, 7.0],
                }
            )
            traits.to_csv(
                root / "Training_data/1_Training_Trait_Data_2014_2023.csv",
                index=False,
            )
            genotype_text = "<Numeric>\n<Marker>\tm1\nH1\t0\nH2\t0.5\nH3\t1\n"
            (root / "Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt").write_text(
                genotype_text
            )
            pd.DataFrame(
                {"Env": ["AA_2024"], "Hybrid": ["H1"], "Yield_Mg_ha": [np.nan]}
            ).to_csv(root / "Testing_data/1_Submission_Template_2024.csv", index=False)
            pd.DataFrame(
                {"Env": ["AA_2024"], "Hybrid": ["H1"], "Yield_Mg_ha": [5.0]}
            ).to_csv(root / "Testing_data/7_Testing_Observed_Values.csv", index=False)

            _, response, _, _, _, facts = load_inputs(root, min_trait_observations=0)

            self.assertTrue(np.isnan(response.loc["H1", "AA_2014"]))
            self.assertEqual(facts["cells_with_missing_plots"], 1)


class ParabraSolverTests(unittest.TestCase):
    def test_dual_and_primal_variance_ridge_predictions_match(self):
        rng = np.random.default_rng(17)
        features = rng.normal(size=(10, 5))
        observed_indices = np.array([0, 1, 3, 4, 6, 8, 9])
        response = rng.normal(size=len(observed_indices))
        gram = features @ features.T

        dual = fit_variance_ridge_dual(
            gram[np.ix_(observed_indices, observed_indices)],
            gram[:, observed_indices],
            response,
        )
        primal = fit_variance_ridge_primal(
            features[observed_indices],
            features,
            response,
        )

        np.testing.assert_allclose(
            dual.predictions + dual.intercept,
            primal.predictions,
            rtol=1e-7,
            atol=1e-7,
        )
        self.assertAlmostEqual(dual.h2, primal.h2, places=7)
        self.assertAlmostEqual(dual.ridge, primal.ridge, places=7)

    def test_spectral_solver_matches_coordinate_fixed_point(self):
        rng = np.random.default_rng(31)
        features = rng.normal(size=(12, 6))
        response = rng.normal(size=12)
        spectral = fit_variance_ridge_primal(
            features,
            features,
            response,
            tolerance=1e-12,
            max_iterations=1000,
        )

        n, marker_count = features.shape
        intercept = float(response.mean())
        y = response - intercept
        tilde = features.T @ y
        centered = features - features.mean(axis=0)
        sums_of_squares = np.square(centered).sum(axis=0)
        trace_xsx = float(sums_of_squares.sum())
        vy = float(y @ y / (n - 1))
        ve = 0.5 * vy
        vb = 0.5 * vy / (trace_xsx / (n - 1))
        ve0 = ve * 20.0
        vb0 = vb * 20.0
        ridge = ve / vb
        coefficients = np.zeros(marker_count)
        residual = y.copy()

        for _ in range(1000):
            old = coefficients.copy()
            for column in range(marker_count):
                value = (
                    residual @ centered[:, column]
                    + sums_of_squares[column] * coefficients[column]
                ) / (sums_of_squares[column] + ridge)
                residual -= centered[:, column] * (value - coefficients[column])
                coefficients[column] = value
            residual -= residual.mean()
            ve = (float(residual @ y) + ve0) / (n + 20.0)
            vb = (float(tilde @ coefficients) + vb0) / (trace_xsx + 20.0)
            ridge = ve / vb
            if np.square(old - coefficients).sum() < 1e-12:
                break

        coordinate_predictions = features @ coefficients + intercept
        np.testing.assert_allclose(
            spectral.predictions,
            coordinate_predictions,
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertAlmostEqual(spectral.h2, 1.0 - ve / vy, places=5)

    def test_full_rank_megasem_accepts_trait_missing_values(self):
        rng = np.random.default_rng(23)
        features = rng.normal(size=(14, 7))
        gram = features @ features.T
        fit_indices = np.arange(11)
        response = np.column_stack(
            [
                features[:11, 0] + rng.normal(scale=0.1, size=11),
                features[:11, 1] + rng.normal(scale=0.1, size=11),
                features[:11, 2] + rng.normal(scale=0.1, size=11),
            ]
        )
        response[[1, 5], 0] = np.nan
        response[[2, 7], 1] = np.nan
        response[[0, 8], 2] = np.nan

        fit = fit_full_rank_megasem(
            response,
            gram,
            fit_indices,
            ["IAH1_2021", "IAH2_2022", "DEH1_2023"],
        )
        states, names = aggregate_state_predictions(
            fit.predictions,
            ["IAH1_2021", "IAH2_2022", "DEH1_2023"],
        )

        self.assertEqual(fit.predictions.shape, (14, 3))
        self.assertTrue(np.isfinite(fit.predictions).all())
        self.assertEqual(states.shape, (14, 2))
        self.assertEqual(names, ["IA", "DE"])
        self.assertEqual(len(fit.first_stage), 3)
        self.assertEqual(len(fit.second_stage), 3)
        json.dumps({"first_stage": fit.first_stage, "second_stage": fit.second_stage})


class ParabraScoreTests(unittest.TestCase):
    def test_score_uses_only_transformer_observed_intersection_for_zscores(self):
        environments = [f"E{index:02d}_2024" for index in range(22)]
        submission_rows = []
        observed_rows = []
        transformer_rows = []
        for environment in environments:
            for index, (actual, parabra, transformer) in enumerate(
                [(1.0, 0.0, 0.0), (2.0, 1.0, 2.0), (3.0, 2.0, 1.0)]
            ):
                hybrid = f"H{index}"
                sample_id = f"{environment}-{hybrid}"
                submission_rows.append((environment, hybrid, parabra))
                observed_rows.append((environment, hybrid, actual))
                transformer_rows.append(
                    (sample_id, environment, hybrid, actual, transformer)
                )
            submission_rows.append((environment, "HX", 20.0))
            transformer_rows.append(
                (f"{environment}-HX", environment, "HX", np.nan, -10.0)
            )

        submission = pd.DataFrame(
            submission_rows,
            columns=["Env", "Hybrid", "Yield_Mg_ha"],
        )
        observed = pd.DataFrame(
            observed_rows,
            columns=["Env", "Hybrid", "Yield_Mg_ha"],
        )
        transformer = pd.DataFrame(
            transformer_rows,
            columns=["id", "Env", "Hybrid", "Actual", "Pred"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            transformer_path = output_dir / "transformer.csv"
            transformer.to_csv(transformer_path, index=False)
            summary = save_scores(output_dir, submission, observed, transformer_path)
            scored = pd.read_csv(output_dir / "ensemble_scored_predictions.csv")

            self.assertEqual(summary["comparison"]["normalization_rows"], 66)
            self.assertEqual(len(scored), 66)
            self.assertFalse((scored["Hybrid"] == "HX").any())
            np.testing.assert_allclose(
                scored.groupby("Env")["ParabraZ"].mean().to_numpy(),
                0.0,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                scored.groupby("Env")["TransformerZ"].mean().to_numpy(),
                0.0,
                atol=1e-7,
            )
            self.assertFalse((output_dir / "ensemble_template_predictions.csv").exists())

    def test_equal_zscore_ensemble_and_macro_pcc(self):
        frame = pd.DataFrame(
            {
                "Env": ["A_2024"] * 3 + ["B_2024"] * 3,
                "Actual": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
                "P1": [2.0, 4.0, 6.0, 9.0, 6.0, 3.0],
                "P2": [1.0, 3.0, 5.0, 6.0, 4.0, 2.0],
            }
        )
        frame["Pred"] = 0.5 * (
            within_environment_zscore(frame, "P1")
            + within_environment_zscore(frame, "P2")
        )
        per_environment, summary = environment_scores(frame, "Pred")

        self.assertEqual(len(per_environment), 2)
        self.assertAlmostEqual(summary["macro_environment_pcc"], 1.0)

    def test_score_fails_for_constant_environment_prediction(self):
        frame = pd.DataFrame(
            {
                "Env": ["A_2024"] * 3,
                "Actual": [1.0, 2.0, 3.0],
                "Pred": [1.0, 1.0, 1.0],
            }
        )
        with self.assertRaises(ValueError):
            environment_scores(frame, "Pred")


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils.canonical_data import (
    CanonicalDataBundle,
    fit_environment_transform,
    fit_hybrid_covariate_transform,
    fit_order_independent_genotypes,
)


class CanonicalDataBundleTests(unittest.TestCase):
    def _write_bundle(self, root: Path) -> None:
        manifest = {
            "schema_version": "canonical-normalized-v1",
            "marker_columns": ["m1", "m2"],
            "environment_feature_sets": {
                "core": {
                    "columns": ["f1", "f2", "constant", "duplicate"],
                    "drop_constant": ["constant"],
                    "drop_exact_duplicates": {"duplicate": "f1"},
                }
            },
            "variants": {
                "repair": {
                    "row_policy": "complete",
                    "window": "planting",
                    "feature_set": "core",
                },
                "recover": {
                    "row_policy": "recovered",
                    "window": "preplant14",
                    "feature_set": "core",
                },
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        pd.DataFrame(
            {
                "Hybrid": ["A/B", "A/C", "D/C"],
                "m1": [0.0, np.nan, 1.0],
                "m2": [np.nan, 2.0, np.nan],
            }
        ).to_csv(root / "genotypes.csv", index=False)
        environments = pd.DataFrame(
            {
                "Env": ["E1_2022", "E2_2023", "E3_2024"],
                "f1": [1.0, 3.0, 5.0],
                "f2": [10.0, np.nan, 30.0],
                "constant": [1.0, 1.0, 1.0],
                "duplicate": [1.0, 3.0, 5.0],
            }
        )
        environments.to_csv(root / "environments_planting.csv", index=False)
        environments.to_csv(root / "environments_preplant14.csv", index=False)
        train = pd.DataFrame(
            {
                "row_index": [0, 1],
                "source_row_index": [10, 11],
                "id": ["E1_2022-A/B", "E2_2023-A/C"],
                "Env": ["E1_2022", "E2_2023"],
                "Hybrid": ["A/B", "A/C"],
                "Yield_Mg_ha": [1.2, 2.3],
            }
        )
        test = pd.DataFrame(
            {
                "row_index": [0],
                "source_row_index": [20],
                "id": ["E3_2024-D/C"],
                "Env": ["E3_2024"],
                "Hybrid": ["D/C"],
                "Yield_Mg_ha": [3.4],
            }
        )
        for policy in ("complete", "recovered"):
            train.to_csv(root / f"rows_{policy}_train.csv", index=False)
            test.to_csv(root / f"rows_{policy}_test.csv", index=False)

    def _add_hybrid_feature_set(self, root: Path) -> None:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["hybrid_feature_sets"] = {
            "parent_public": {
                "columns": [
                    "anthesis_gdd",
                    "coverage",
                    "h_constant",
                    "h_duplicate",
                ],
                "drop_constant": ["h_constant"],
                "drop_exact_duplicates": {"h_duplicate": "anthesis_gdd"},
            }
        }
        manifest["variants"]["repair_parent_public"] = {
            "row_policy": "complete",
            "window": "planting",
            "feature_set": "core",
            "hybrid_feature_set": "parent_public",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        pd.DataFrame(
            {
                "Hybrid": ["A/B", "A/C", "D/C"],
                "anthesis_gdd": [100.0, np.nan, 130.0],
                "coverage": [1.0, 0.0, 1.0],
                "h_constant": [1.0, 1.0, 1.0],
                "h_duplicate": [100.0, np.nan, 130.0],
            }
        ).to_csv(root / "hybrid_covariates.csv", index=False)

    def test_loads_variant_and_exposes_fingerprints_and_ordered_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_bundle(root)
            bundle = CanonicalDataBundle(root)
            view = bundle.load_variant("repair", "train")

            self.assertEqual(bundle.variant_names, ("repair", "recover"))
            self.assertEqual(view.marker_columns, ("m1", "m2"))
            self.assertEqual(
                view.environment_feature_columns,
                ("f1", "f2", "constant", "duplicate"),
            )
            self.assertEqual(view.environment_columns, ("f1", "f2"))
            self.assertIsNone(view.hybrid_feature_set)
            self.assertIsNone(view.hybrid_covariates)
            self.assertEqual(view.hybrid_feature_columns, ())
            self.assertEqual(view.hybrid_columns, ())
            self.assertEqual(view.rows["row_index"].tolist(), [0, 1])
            self.assertRegex(view.manifest_fingerprint, r"^[0-9a-f]{64}$")
            self.assertRegex(view.dataset_fingerprint, r"^[0-9a-f]{64}$")
            self.assertEqual(
                view.dataset_fingerprint,
                bundle.load_variant("repair", "train").dataset_fingerprint,
            )
            self.assertNotEqual(
                view.dataset_fingerprint,
                bundle.load_variant("repair", "test").dataset_fingerprint,
            )

    def test_hybrid_feature_set_is_explicit_and_changes_only_selected_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_bundle(root)
            self._add_hybrid_feature_set(root)
            bundle = CanonicalDataBundle(root)
            control = bundle.load_variant("repair", "train")
            treatment = bundle.load_variant("repair_parent_public", "train")

            self.assertIsNone(control.hybrid_feature_set)
            self.assertEqual(control.hybrid_columns, ())
            self.assertEqual(treatment.hybrid_feature_set, "parent_public")
            self.assertEqual(
                treatment.hybrid_feature_columns,
                ("anthesis_gdd", "coverage", "h_constant", "h_duplicate"),
            )
            self.assertEqual(
                treatment.hybrid_columns, ("anthesis_gdd", "coverage")
            )
            self.assertEqual(treatment.marker_columns, control.marker_columns)
            self.assertEqual(
                treatment.hybrid_covariates["Hybrid"].tolist(),
                ["A/B", "A/C", "D/C"],
            )
            self.assertNotEqual(
                treatment.dataset_fingerprint, control.dataset_fingerprint
            )
            transformed = fit_hybrid_covariate_transform(
                treatment, ["A/B", "D/C"]
            )
            self.assertEqual(
                transformed.values.columns.tolist(),
                ["Hybrid", "anthesis_gdd", "coverage"],
            )
            self.assertEqual(
                transformed.provenance["train_reference_hybrid_count"], 2
            )

            control_fingerprint = control.dataset_fingerprint
            treatment_fingerprint = treatment.dataset_fingerprint
            sidecar = pd.read_csv(root / "hybrid_covariates.csv")
            sidecar.loc[0, "anthesis_gdd"] = 101.0
            sidecar.to_csv(root / "hybrid_covariates.csv", index=False)
            refreshed = CanonicalDataBundle(root)
            self.assertEqual(
                refreshed.load_variant("repair", "train").dataset_fingerprint,
                control_fingerprint,
            )
            self.assertNotEqual(
                refreshed.load_variant(
                    "repair_parent_public", "train"
                ).dataset_fingerprint,
                treatment_fingerprint,
            )

    def test_hybrid_feature_set_fails_closed_on_unknown_set_or_bad_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_bundle(root)
            self._add_hybrid_feature_set(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["variants"]["repair_parent_public"][
                "hybrid_feature_set"
            ] = "unknown"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "unknown hybrid feature sets"
            ):
                CanonicalDataBundle(root)

            self._write_bundle(root)
            self._add_hybrid_feature_set(root)
            sidecar_path = root / "hybrid_covariates.csv"
            sidecar = pd.read_csv(sidecar_path).iloc[:-1]
            sidecar.to_csv(sidecar_path, index=False)
            with self.assertRaisesRegex(ValueError, "must exactly match"):
                CanonicalDataBundle(root).load_variant(
                    "repair_parent_public", "train"
                )

            self._write_bundle(root)
            self._add_hybrid_feature_set(root)
            sidecar = pd.read_csv(sidecar_path).drop(columns="coverage")
            sidecar.to_csv(sidecar_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing feature-set columns"):
                CanonicalDataBundle(root).load_variant(
                    "repair_parent_public", "train"
                )

            self._write_bundle(root)
            self._add_hybrid_feature_set(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["hybrid_feature_sets"]["parent_public"] = {
                "columns": ["m1"]
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            sidecar = pd.read_csv(sidecar_path)
            sidecar["m1"] = [1.0, 2.0, 3.0]
            sidecar.to_csv(sidecar_path, index=False)
            with self.assertRaisesRegex(ValueError, "collide with marker"):
                CanonicalDataBundle(root).load_variant(
                    "repair_parent_public", "train"
                )

    def test_rows_fail_closed_on_noncontiguous_index_or_unknown_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_bundle(root)
            path = root / "rows_complete_train.csv"
            rows = pd.read_csv(path)
            rows.loc[1, "row_index"] = 7
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "row_index must be contiguous"):
                CanonicalDataBundle(root).load_variant("repair", "train")

            self._write_bundle(root)
            rows = pd.read_csv(path)
            rows.loc[1, "Hybrid"] = "X/Y"
            rows.loc[1, "id"] = "E2_2023-X/Y"
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "absent genotype entities"):
                CanonicalDataBundle(root).load_variant("repair", "train")

    def test_rows_fail_closed_when_id_and_hybrid_disagree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_bundle(root)
            path = root / "rows_complete_train.csv"
            rows = pd.read_csv(path)
            rows.loc[0, "id"] = "E1_2022-A/C"
            rows.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Hybrid does not match"):
                CanonicalDataBundle(root).load_variant("repair", "train")


class CanonicalTransformTests(unittest.TestCase):
    def test_genotype_imputation_uses_original_reference_calls_and_is_order_independent(self):
        genotypes = pd.DataFrame(
            {
                "Hybrid": ["A/B", "A/C", "D/C", "B/C", "X/Y"],
                "m1": [0.0, np.nan, 1.0, np.nan, np.nan],
                "m2": [np.nan, 2.0, np.nan, np.nan, np.nan],
            }
        )
        reference = ["A/B", "A/C", "D/C"]
        result = fit_order_independent_genotypes(genotypes, reference, ["m1", "m2"])
        by_hybrid = result.values.set_index("Hybrid")

        # B/C sees the union of original A/B, A/C, and D/C sibling calls.
        self.assertEqual(by_hybrid.loc["B/C", "m1"], 0.5)
        self.assertEqual(by_hybrid.loc["B/C", "m2"], 2.0)
        # X/Y has no siblings and therefore receives reference marker medians.
        self.assertEqual(by_hybrid.loc["X/Y", "m1"], 0.5)
        self.assertEqual(by_hybrid.loc["X/Y", "m2"], 2.0)
        self.assertTrue(result.mask.set_index("Hybrid").loc["B/C", "m1"])
        self.assertEqual(result.provenance["original_missing_count"], 7)
        self.assertTrue(result.provenance["frozen_marker_selection"])

        reordered = genotypes.iloc[::-1].reset_index(drop=True)
        result_reordered = fit_order_independent_genotypes(
            reordered, list(reversed(reference)), ["m1", "m2"]
        )
        pd.testing.assert_frame_equal(
            result.values.sort_values("Hybrid").reset_index(drop=True),
            result_reordered.values.sort_values("Hybrid").reset_index(drop=True),
        )

    def test_genotype_imputation_fails_without_reference_fallback(self):
        genotypes = pd.DataFrame(
            {"Hybrid": ["A/B", "A/C"], "m1": [0.0, 1.0], "m2": [np.nan, np.nan]}
        )
        with self.assertRaisesRegex(ValueError, "no observed fallback"):
            fit_order_independent_genotypes(genotypes, ["A/B", "A/C"], ["m1", "m2"])

    def test_environment_transform_uses_unique_env_fit_and_manifest_drops(self):
        environments = pd.DataFrame(
            {
                "Env": ["E1", "E2", "E3"],
                "f1": [1.0, 3.0, 5.0],
                "f2": [10.0, np.nan, 30.0],
                "constant": [1.0, 1.0, 1.0],
                "duplicate": [1.0, 3.0, 5.0],
            }
        )
        result = fit_environment_transform(
            environments,
            ["E1", "E1", "E2"],
            ["f1", "f2", "constant", "duplicate"],
            constant_columns=["constant"],
            exact_duplicate_columns=["duplicate"],
        )

        self.assertEqual(result.values.columns.tolist(), ["Env", "f1", "f2"])
        self.assertEqual(result.provenance["train_reference_env_count"], 2)
        self.assertEqual(result.provenance["scaler_n_samples_seen"], 2)
        self.assertEqual(result.provenance["medians"], {"f1": 2.0, "f2": 10.0})
        self.assertTrue(result.mask.set_index("Env").loc["E2", "f2"])
        transformed = result.values.set_index("Env")
        self.assertAlmostEqual(transformed.loc["E1", "f1"], -1.0)
        self.assertAlmostEqual(transformed.loc["E2", "f1"], 1.0)
        self.assertAlmostEqual(transformed.loc["E2", "f2"], 0.0)

    def test_environment_transform_fails_without_training_median(self):
        environments = pd.DataFrame(
            {"Env": ["E1", "E2"], "f1": [np.nan, 2.0]}
        )
        with self.assertRaisesRegex(ValueError, "no observed median"):
            fit_environment_transform(environments, ["E1"], ["f1"])

    def test_environment_transform_neutralizes_train_unobserved_feature(self):
        environments = pd.DataFrame(
            {
                "Env": ["E1", "E2", "E3"],
                "observed": [1.0, 3.0, 9.0],
                "future_only": [np.nan, np.nan, 17.0],
            }
        )
        result = fit_environment_transform(
            environments,
            ["E1", "E2"],
            ["observed", "future_only"],
            unobserved_reference_policy="neutralize",
        )

        transformed = result.values.set_index("Env")
        self.assertEqual(
            transformed["future_only"].tolist(),
            [0.0, 0.0, 0.0],
        )
        self.assertTrue(
            result.mask.set_index("Env")["future_only"].all()
        )
        self.assertEqual(
            result.provenance["neutralized_unobserved_columns"],
            ["future_only"],
        )
        self.assertEqual(
            result.provenance["algorithm"],
            "train_unique_env_median_standard_scaler_neutralize_unobserved_v1",
        )

    def test_environment_transform_rejects_unknown_unobserved_policy(self):
        environments = pd.DataFrame(
            {"Env": ["E1"], "f1": [1.0]}
        )
        with self.assertRaisesRegex(ValueError, "unobserved_reference_policy"):
            fit_environment_transform(
                environments,
                ["E1"],
                ["f1"],
                unobserved_reference_policy="invent",
            )

    def test_hybrid_covariate_transform_uses_unique_training_hybrids_and_drops(self):
        covariates = pd.DataFrame(
            {
                "Hybrid": ["A/B", "A/C", "D/C"],
                "anthesis_gdd": [1.0, 3.0, 5.0],
                "height": [10.0, np.nan, 30.0],
                "constant": [1.0, 1.0, 1.0],
                "duplicate": [1.0, 3.0, 5.0],
            }
        )
        result = fit_hybrid_covariate_transform(
            covariates,
            ["A/B", "A/B", "A/C"],
            ["anthesis_gdd", "height", "constant", "duplicate"],
            constant_columns=["constant"],
            exact_duplicate_columns=["duplicate"],
        )

        self.assertEqual(
            result.values.columns.tolist(), ["Hybrid", "anthesis_gdd", "height"]
        )
        self.assertEqual(result.provenance["train_reference_hybrid_count"], 2)
        self.assertEqual(result.provenance["scaler_n_samples_seen"], 2)
        self.assertEqual(
            result.provenance["medians"],
            {"anthesis_gdd": 2.0, "height": 10.0},
        )
        self.assertEqual(
            result.provenance["reference_observed_count"],
            {"anthesis_gdd": 2, "height": 1},
        )
        self.assertTrue(result.mask.set_index("Hybrid").loc["A/C", "height"])
        transformed = result.values.set_index("Hybrid")
        self.assertAlmostEqual(transformed.loc["A/B", "anthesis_gdd"], -1.0)
        self.assertAlmostEqual(transformed.loc["A/C", "anthesis_gdd"], 1.0)
        self.assertAlmostEqual(transformed.loc["A/C", "height"], 0.0)
        # D/C is transformed by training-only statistics, not refit with its value.
        self.assertAlmostEqual(transformed.loc["D/C", "height"], 20.0)

    def test_hybrid_covariate_transform_fails_without_training_median(self):
        covariates = pd.DataFrame(
            {
                "Hybrid": ["A/B", "A/C", "D/C"],
                "anthesis_gdd": [np.nan, np.nan, 5.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "no observed median"):
            fit_hybrid_covariate_transform(
                covariates,
                ["A/B", "A/C"],
                ["anthesis_gdd"],
            )


if __name__ == "__main__":
    unittest.main()

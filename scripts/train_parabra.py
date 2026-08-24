#!/usr/bin/env python
"""Fit PARaBra on 2014-2023 and score its fixed 2024 ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.parabra import (  # noqa: E402
    aggregate_state_predictions,
    author_gaussian_kernel,
    environment_scores,
    fixed_equal_zscore_ensemble,
    fit_full_rank_megasem,
    kernel_feature_gram,
    mean_impute_markers,
)


TRAIT_RELATIVE_PATH = Path("Training_data/1_Training_Trait_Data_2014_2023.csv")
GENOTYPE_RELATIVE_PATH = Path(
    "Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt"
)
TEMPLATE_RELATIVE_PATH = Path("Testing_data/1_Submission_Template_2024.csv")
OBSERVED_RELATIVE_PATH = Path("Testing_data/7_Testing_Observed_Values.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the Python PARaBra model and its fixed transformer ensemble."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--transformer-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--min-trait-observations", type=int, default=100)
    return parser.parse_args()


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_facts() -> dict[str, str | list[str]]:
    def run(*parts: str) -> str:
        result = subprocess.run(
            ["git", *parts],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": status.splitlines() if status else [],
    }


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required files do not exist: {missing}")


def load_inputs(
    data_root: Path,
    min_trait_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    trait_path = data_root / TRAIT_RELATIVE_PATH
    genotype_path = data_root / GENOTYPE_RELATIVE_PATH
    template_path = data_root / TEMPLATE_RELATIVE_PATH
    observed_path = data_root / OBSERVED_RELATIVE_PATH
    require_paths([trait_path, genotype_path, template_path, observed_path])

    log("Read the genotype matrix.")
    genotype = pd.read_csv(
        genotype_path,
        sep="\t",
        skiprows=1,
        index_col=0,
        na_values=["NA"],
    )
    genotype.index = genotype.index.astype(str)
    genotype = genotype.astype(np.float32, copy=False)
    if not genotype.index.is_unique:
        raise ValueError("The genotype hybrid names are not unique.")

    log("Read the trait and 2024 files.")
    traits = pd.read_csv(
        trait_path,
        usecols=["Env", "Year", "Hybrid", "Yield_Mg_ha"],
    )
    template = pd.read_csv(template_path)
    observed = pd.read_csv(observed_path)
    for name, frame in (("template", template), ("observed", observed)):
        required = {"Env", "Hybrid", "Yield_Mg_ha"}
        if not required.issubset(frame.columns):
            raise ValueError(f"The {name} file lacks columns: {sorted(required - set(frame.columns))}")
        if frame.duplicated(["Env", "Hybrid"]).any():
            raise ValueError(f"The {name} file has duplicate Env and Hybrid rows.")

    years = set(pd.to_numeric(traits["Year"], errors="raise").astype(int).unique())
    if min(years) != 2014 or max(years) != 2023:
        raise ValueError(f"The trait years do not span 2014-2023: {sorted(years)}")
    if not template["Env"].astype(str).str.endswith("2024").all():
        raise ValueError("The template contains a non-2024 environment.")

    target_states = set(template["Env"].astype(str).str[:2])
    traits["State"] = traits["Env"].astype(str).str[:2]
    traits = traits.loc[traits["State"].isin(target_states)].copy()
    candidate_traits = int(traits["Env"].nunique())
    grouped_yield = traits.groupby(["Hybrid", "Env"], sort=True)["Yield_Mg_ha"]
    plot_means = grouped_yield.mean()
    cells_with_missing_plots = grouped_yield.count() < grouped_yield.size()
    plot_means.loc[cells_with_missing_plots] = np.nan
    response = plot_means.unstack("Env")
    response = response.loc[:, response.notna().sum(axis=0) > min_trait_observations]
    response = (response - response.mean(axis=0)) / response.std(axis=0, ddof=1)
    if response.empty or response.isna().all(axis=0).any():
        raise ValueError("The response matrix has no usable environment traits.")

    overlap = response.index[response.index.isin(genotype.index)]
    response = response.loc[overlap]
    if response.empty:
        raise ValueError("The response and genotype matrices have no hybrid overlap.")
    if not set(template["Hybrid"].astype(str)).issubset(set(genotype.index)):
        raise ValueError("A 2024 hybrid has no genotype row.")

    facts = {
        "genotype_hybrids": int(genotype.shape[0]),
        "markers": int(genotype.shape[1]),
        "candidate_environment_traits": candidate_traits,
        "retained_environment_traits": int(response.shape[1]),
        "fit_hybrids": int(response.shape[0]),
        "observed_fit_cells": int(response.notna().sum().sum()),
        "cells_with_missing_plots": int(cells_with_missing_plots.sum()),
        "template_rows": int(len(template)),
        "observed_rows": int(len(observed)),
        "target_states": int(len(target_states)),
    }
    return genotype, response, template, observed, traits, facts


def save_scores(
    output_dir: Path,
    submission: pd.DataFrame,
    observed: pd.DataFrame,
    transformer_path: Path,
) -> dict[str, dict[str, float | int]]:
    expected_environments = int(observed["Env"].nunique())
    if expected_environments != 22:
        raise ValueError(f"The scored set has {expected_environments} environments, not 22.")

    parabra_predictions = submission[["Env", "Hybrid", "Yield_Mg_ha"]].rename(
        columns={"Yield_Mg_ha": "ParabraPred"}
    )
    parabra_predictions.insert(
        0,
        "id",
        parabra_predictions["Env"].astype(str)
        + "-"
        + parabra_predictions["Hybrid"].astype(str),
    )

    observed_predictions = observed[["Env", "Hybrid", "Yield_Mg_ha"]].rename(
        columns={"Yield_Mg_ha": "Actual"}
    )
    observed_predictions.insert(
        0,
        "id",
        observed_predictions["Env"].astype(str)
        + "-"
        + observed_predictions["Hybrid"].astype(str),
    )

    transformer = pd.read_csv(transformer_path)
    required = {"id", "Env", "Hybrid", "Actual", "Pred"}
    if not required.issubset(transformer.columns):
        raise ValueError(
            f"The transformer file lacks columns: {sorted(required - set(transformer.columns))}"
        )
    if transformer.duplicated(["Env", "Hybrid"]).any() or transformer["id"].duplicated().any():
        raise ValueError("The transformer prediction keys are not unique.")

    keep = transformer[["id", "Env", "Hybrid", "Actual", "Pred"]].rename(
        columns={"Actual": "TransformerActual", "Pred": "TransformerPred"}
    )
    scored = observed_predictions.merge(
        keep,
        on=["id", "Env", "Hybrid"],
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != len(observed_predictions):
        raise ValueError("A 2024 observed row has no transformer prediction.")
    if not np.allclose(
        scored["Actual"].to_numpy(dtype=float),
        scored["TransformerActual"].to_numpy(dtype=float),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("The observed and transformer target values differ.")

    scored = scored.merge(
        parabra_predictions,
        on=["id", "Env", "Hybrid"],
        how="left",
        validate="one_to_one",
    )
    if scored["ParabraPred"].isna().any() or scored["TransformerPred"].isna().any():
        raise ValueError("A common 2024 row has no model prediction.")
    scored = fixed_equal_zscore_ensemble(
        scored,
        parabra_column="ParabraPred",
        transformer_column="TransformerPred",
    )

    parabra_env, parabra_summary = environment_scores(scored, "ParabraPred")
    transformer_env, transformer_summary = environment_scores(scored, "TransformerPred")
    ensemble_env, ensemble_summary = environment_scores(scored, "EnsemblePred")
    summaries_to_check = [parabra_summary, transformer_summary, ensemble_summary]
    if any(item["environments"] != expected_environments for item in summaries_to_check):
        raise ValueError("A score report does not contain all 22 environments.")

    parabra_env.to_csv(output_dir / "parabra_environment_metrics.csv", index=False)
    scored[
        ["id", "Env", "Hybrid", "Actual", "ParabraPred"]
    ].rename(columns={"ParabraPred": "Pred"}).assign(
        model_name="parabra_python_zsemf"
    ).to_csv(output_dir / "parabra_scored_predictions.csv", index=False)
    scored[
        [
            "id",
            "Env",
            "Hybrid",
            "Actual",
            "ParabraPred",
            "TransformerPred",
            "ParabraZ",
            "TransformerZ",
            "EnsemblePred",
        ]
    ].to_csv(output_dir / "ensemble_scored_predictions.csv", index=False)

    comparison = parabra_env.rename(
        columns={"pcc": "parabra_pcc"}
    ).merge(
        transformer_env[["Env", "pcc"]].rename(
            columns={"pcc": "transformer_pcc"}
        ),
        on="Env",
        validate="one_to_one",
    ).merge(
        ensemble_env[["Env", "pcc"]].rename(
            columns={"pcc": "ensemble_pcc"}
        ),
        on="Env",
        validate="one_to_one",
    )
    comparison["ensemble_minus_transformer"] = (
        comparison["ensemble_pcc"] - comparison["transformer_pcc"]
    )
    comparison["ensemble_minus_parabra"] = (
        comparison["ensemble_pcc"] - comparison["parabra_pcc"]
    )
    comparison.to_csv(output_dir / "ensemble_environment_metrics.csv", index=False)

    summaries = {
        "parabra": parabra_summary,
        "transformer": transformer_summary,
        "equal_zscore_ensemble": ensemble_summary,
        "comparison": {
            "competition_target": 0.437,
            "parabra_minus_target": (
                parabra_summary["macro_environment_pcc"] - 0.437
            ),
            "ensemble_minus_target": (
                ensemble_summary["macro_environment_pcc"] - 0.437
            ),
            "parabra_minus_transformer": (
                parabra_summary["macro_environment_pcc"]
                - transformer_summary["macro_environment_pcc"]
            ),
            "ensemble_minus_transformer": (
                ensemble_summary["macro_environment_pcc"]
                - transformer_summary["macro_environment_pcc"]
            ),
            "parabra_environment_wins": int(
                (comparison["parabra_pcc"] > comparison["transformer_pcc"]).sum()
            ),
            "ensemble_environment_wins": int(
                (comparison["ensemble_pcc"] > comparison["transformer_pcc"]).sum()
            ),
            "normalization_rows": int(len(scored)),
            "normalization_environments": expected_environments,
        },
    }
    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(summaries, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summaries


def main() -> None:
    args = parse_args()
    start = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    trait_path = args.data_root / TRAIT_RELATIVE_PATH
    genotype_path = args.data_root / GENOTYPE_RELATIVE_PATH
    template_path = args.data_root / TEMPLATE_RELATIVE_PATH
    observed_path = args.data_root / OBSERVED_RELATIVE_PATH
    require_paths(
        [
            trait_path,
            genotype_path,
            template_path,
            observed_path,
            args.transformer_predictions,
        ]
    )

    genotype, response, template, observed, _, data_facts = load_inputs(
        args.data_root,
        args.min_trait_observations,
    )
    log(
        "Use "
        f"{data_facts['fit_hybrids']} fit hybrids and "
        f"{data_facts['retained_environment_traits']} environment traits."
    )

    marker_values, missing_markers = mean_impute_markers(genotype.to_numpy(copy=False))
    log(f"Replace {missing_markers} missing marker values with marker means.")
    log("Create the author EigenGAU kernel.")
    kernel, mean_distance = author_gaussian_kernel(marker_values, phi=args.phi)
    del marker_values
    log(f"The mean marker distance is {mean_distance:.8f}.")

    log("Create the K2X feature Gram matrix as K times K.")
    feature_gram = kernel_feature_gram(kernel)
    del kernel

    genotype_lookup = pd.Series(np.arange(len(genotype), dtype=np.int64), index=genotype.index)
    fit_indices = genotype_lookup.loc[response.index].to_numpy(dtype=np.int64)
    log("Fit the full-rank ZSEMF model.")
    fit = fit_full_rank_megasem(
        response.to_numpy(dtype=np.float64),
        feature_gram,
        fit_indices,
        response.columns.astype(str).tolist(),
        progress=log,
    )
    del feature_gram

    state_predictions, states = aggregate_state_predictions(
        fit.predictions,
        response.columns.astype(str).tolist(),
    )
    hybrid_indices = genotype.index.get_indexer(template["Hybrid"].astype(str))
    state_indices = pd.Index(states).get_indexer(template["Env"].astype(str).str[:2])
    if (hybrid_indices < 0).any() or (state_indices < 0).any():
        raise ValueError("A template row has no hybrid or state prediction.")

    submission = template[["Env", "Hybrid", "Yield_Mg_ha"]].copy()
    submission["Yield_Mg_ha"] = state_predictions[hybrid_indices, state_indices]
    submission.to_csv(args.output_dir / "parabra_submission.csv", index=False)

    np.savez_compressed(
        args.output_dir / "parabra_fit_predictions.npz",
        hybrids=genotype.index.to_numpy(dtype=str),
        environments=response.columns.to_numpy(dtype=str),
        environment_predictions=fit.predictions,
        states=np.asarray(states, dtype=str),
        state_predictions=state_predictions,
    )
    state_frame = pd.DataFrame(state_predictions, index=genotype.index, columns=states)
    state_frame.index.name = "Hybrid"
    state_frame.to_csv(args.output_dir / "parabra_state_predictions.csv")

    log("Score PARaBra and the fixed equal transformer ensemble.")
    summaries = save_scores(
        args.output_dir,
        submission,
        observed,
        args.transformer_predictions,
    )

    output_paths = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - start,
        "method": {
            "name": "PARaBra full-rank MegaSEM",
            "python_model": "spectral fixed-point form of bWGR ZSEMF",
            "training_years": [2014, 2023],
            "response": (
                "mean yield for each Hybrid and Env, with R mean NA behavior, "
                "then sample z-score by Env"
            ),
            "trait_filter": f"more than {args.min_trait_observations} observed hybrids",
            "marker_imputation": "marker mean across all genotype rows",
            "kernel": "exp(-phi * Euclidean_distance / mean_off_diagonal_distance)",
            "phi": args.phi,
            "k2x_equivalence": "Q Q^T equals K K^T when Q is U times singular values",
            "projection": "mean fitted environment score by the first two Env characters",
            "ensemble": (
                "fixed 0.5 mean of within-Env z-scores on the exact intersection "
                "of the observed rows and frozen transformer predictions"
            ),
            "kernel_boundary": (
                "The manuscript prints exp(-D^2/mean(D)). "
                "The author EigenGAU code uses exp(-phi*D/mean(D)). "
                "This run uses the author code with phi=1."
            ),
            "excluded_later_extensions": [
                "XSEMF with npc=10",
                "phi=0.75",
                "MRR3F residual model",
                "state clusters",
            ],
        },
        "data": {**data_facts, "missing_marker_values": missing_markers},
        "fit": {
            "mean_marker_distance": mean_distance,
            "singular_values": fit.singular_values.tolist(),
            "first_stage": fit.first_stage,
            "second_stage": fit.second_stage,
        },
        "scores": summaries,
        "inputs": {
            str(path): sha256_file(path)
            for path in [
                trait_path,
                genotype_path,
                template_path,
                observed_path,
                args.transformer_predictions,
            ]
        },
        "outputs": {path.name: sha256_file(path) for path in output_paths},
        "code": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in [
                REPO_ROOT / "utils/parabra.py",
                REPO_ROOT / "scripts/train_parabra.py",
                REPO_ROOT / "parabra.slurm",
            ]
        },
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        },
        "git": git_facts(),
        "sources": {
            "author_2024_code": (
                "https://github.com/alenxav/G2F/blob/"
                "ad76bfcaa8565e5171ec093ae8582d556181d2c6/demo/megasem.R"
            ),
            "bwgr_source": (
                "https://github.com/alenxav/bWGR/blob/"
                "7ea65a5f3e099b6381111160699229a8e887a480/"
                "src/RcppEigen20230423.cpp"
            ),
        },
    }
    with (args.output_dir / "run_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    log(
        "PARaBra macro PCC: "
        f"{summaries['parabra']['macro_environment_pcc']:.9f}"
    )
    log(
        "Transformer macro PCC: "
        f"{summaries['transformer']['macro_environment_pcc']:.9f}"
    )
    log(
        "Equal ensemble macro PCC: "
        f"{summaries['equal_zscore_ensemble']['macro_environment_pcc']:.9f}"
    )
    log(f"Write all results to {args.output_dir}.")


if __name__ == "__main__":
    main()

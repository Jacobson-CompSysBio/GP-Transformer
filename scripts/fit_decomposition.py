#!/usr/bin/env python3
"""
Fit additive fixed-effect decomposition on training data for SINN-style training.

Computes:
    y_ij = mu + G_i + E_j + GE_ij

where:
    mu   = grand mean of training yield
    G_i  = mean(y for hybrid i) - mu          (genotype main effect)
    E_j  = mean(y for environment j) - mu     (environment main effect)
    GE_ij = y_ij - mu - G_i - E_j             (interaction residual)

Fixed effects (no shrinkage) — matches SINN paper (Potze et al.) design.

Output: JSON file with mu, per-hybrid G_i, per-environment E_j.
The interaction residual GE_ij is computed on the fly per row.

Usage:
    python scripts/fit_decomposition.py [--data-path DATA_PATH] [--train-year-max YEAR] [--output-path PATH]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _env_year(env_str: str) -> int:
    """Extract 4-digit year from environment string like 'DEH1_2014'."""
    for part in str(env_str).split("_"):
        if len(part) == 4 and part.isdigit():
            return int(part)
    raise ValueError(f"Cannot extract year from Env='{env_str}'")


def fit_decomposition(
    data_path: str = "data/maize_data_2014-2023_vs_2024_v2/",
    train_year_max: int = 2022,
    min_obs_per_hybrid: int = 1,
) -> dict:
    """
    Fit the additive decomposition y = mu + G + E + GE on training rows.

    Args:
        data_path: Path to directory containing X_train.csv and y_train.csv
        train_year_max: Maximum year to include in training (default: 2022)
        min_obs_per_hybrid: Minimum observations per hybrid (for diagnostics only;
                            all hybrids are kept in the decomposition)

    Returns:
        dict with keys: mu, G (hybrid->effect), E (env->effect), diagnostics
    """
    # Load metadata and yield
    x_meta = pd.read_csv(os.path.join(data_path, "X_train.csv"), usecols=["id", "Env"])
    y_data = pd.read_csv(os.path.join(data_path, "y_train.csv"))

    df = x_meta.merge(y_data, on="id")
    df["Year"] = df["Env"].apply(_env_year)
    df["Hybrid"] = df["id"].str.split("-", n=1).str[1]
    df["Parent2"] = df["Hybrid"].str.split("/").str[-1]

    # Filter to training years only
    train_mask = df["Year"] <= train_year_max
    train = df[train_mask].copy()
    val = df[~train_mask].copy()

    print(f"Training rows: {len(train):,} (years <= {train_year_max})")
    print(f"Validation rows: {len(val):,} (years > {train_year_max})")
    print(f"Training environments: {train['Env'].nunique()}")
    print(f"Training hybrids: {train['Hybrid'].nunique()}")

    # --- Fixed-effect decomposition ---
    mu = float(train["Yield_Mg_ha"].mean())

    # Per-environment mean effect
    env_means = train.groupby("Env")["Yield_Mg_ha"].mean()
    E = (env_means - mu).to_dict()

    # Per-hybrid mean effect
    hybrid_means = train.groupby("Hybrid")["Yield_Mg_ha"].mean()
    G = (hybrid_means - mu).to_dict()

    # Per-row interaction residual (for diagnostics)
    train = train.copy()
    train["G_hat"] = train["Hybrid"].map(G)
    train["E_hat"] = train["Env"].map(E)
    train["GE_hat"] = train["Yield_Mg_ha"] - mu - train["G_hat"] - train["E_hat"]

    # --- Variance decomposition ---
    ss_total = float(train["Yield_Mg_ha"].var())
    ss_env = float(train["E_hat"].var())
    ss_gen = float(train["G_hat"].var())
    ss_ge = float(train["GE_hat"].var())
    # Cross-terms (should be ~0 for balanced; nonzero for unbalanced)
    ss_cross = ss_total - ss_env - ss_gen - ss_ge

    print(f"\n--- Variance Decomposition ---")
    print(f"Total variance:      {ss_total:.4f}")
    print(f"Environment (E):     {ss_env:.4f} ({100*ss_env/ss_total:.1f}%)")
    print(f"Genotype (G):        {ss_gen:.4f} ({100*ss_gen/ss_total:.1f}%)")
    print(f"Interaction (GE):    {ss_ge:.4f} ({100*ss_ge/ss_total:.1f}%)")
    print(f"Cross-terms:         {ss_cross:.4f} ({100*ss_cross/ss_total:.1f}%)")

    # --- Hybrid observation counts ---
    hcounts = train.groupby("Hybrid").size()
    n_low = int((hcounts < min_obs_per_hybrid).sum()) if min_obs_per_hybrid > 1 else 0
    print(f"\n--- Hybrid Observation Counts ---")
    print(f"Min: {hcounts.min()}, Median: {hcounts.median():.0f}, "
          f"Mean: {hcounts.mean():.1f}, Max: {hcounts.max()}")
    print(f"Hybrids with 1 obs: {(hcounts == 1).sum()}")
    print(f"Hybrids with >= 3 obs: {(hcounts >= 3).sum()} / {len(hcounts)}")
    if n_low > 0:
        print(f"Hybrids below min_obs={min_obs_per_hybrid}: {n_low}")

    # --- Tester (Parent2) distribution ---
    tester_counts = train.groupby("Parent2").size().sort_values(ascending=False)
    print(f"\n--- Tester (Parent2) Distribution ---")
    for tester, count in tester_counts.head(10).items():
        n_hybrids = train[train["Parent2"] == tester]["Hybrid"].nunique()
        print(f"  {tester}: {count:,} rows, {n_hybrids} unique hybrids")
    print(f"  ... {len(tester_counts)} total testers")

    # --- Validation coverage ---
    if len(val) > 0:
        val_hybrids = set(val["Hybrid"].unique())
        train_hybrids = set(train["Hybrid"].unique())
        novel_in_val = val_hybrids - train_hybrids
        print(f"\n--- Validation Year Coverage ---")
        print(f"Val hybrids: {len(val_hybrids)}, "
              f"Novel (no G_hat): {len(novel_in_val)} ({100*len(novel_in_val)/len(val_hybrids):.1f}%)")
        val_envs = set(val["Env"].unique())
        novel_envs = val_envs - set(train["Env"].unique())
        print(f"Val environments: {len(val_envs)}, "
              f"Novel (no E_hat): {len(novel_envs)} ({100*len(novel_envs)/len(val_envs):.1f}%)")

    # --- G_hat statistics ---
    g_values = np.array(list(G.values()))
    e_values = np.array(list(E.values()))
    print(f"\n--- Effect Statistics ---")
    print(f"mu = {mu:.4f}")
    print(f"G_hat: mean={g_values.mean():.4f}, std={g_values.std():.4f}, "
          f"range=[{g_values.min():.4f}, {g_values.max():.4f}]")
    print(f"E_hat: mean={e_values.mean():.4f}, std={e_values.std():.4f}, "
          f"range=[{e_values.min():.4f}, {e_values.max():.4f}]")

    diagnostics = {
        "train_rows": len(train),
        "val_rows": len(val),
        "n_envs": int(train["Env"].nunique()),
        "n_hybrids": int(train["Hybrid"].nunique()),
        "n_testers": int(train["Parent2"].nunique()),
        "train_year_max": train_year_max,
        "variance_total": ss_total,
        "variance_env_frac": ss_env / ss_total,
        "variance_gen_frac": ss_gen / ss_total,
        "variance_ge_frac": ss_ge / ss_total,
        "hybrid_obs_min": int(hcounts.min()),
        "hybrid_obs_median": float(hcounts.median()),
        "hybrids_with_1_obs": int((hcounts == 1).sum()),
    }

    return {
        "mu": mu,
        "G": G,
        "E": E,
        "diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit additive decomposition for SINN-style training")
    parser.add_argument("--data-path", type=str, default="data/maize_data_2014-2023_vs_2024_v2/",
                        help="Path to data directory with X_train.csv and y_train.csv")
    parser.add_argument("--train-year-max", type=int, default=2022,
                        help="Maximum year to include in training decomposition")
    parser.add_argument("--output-path", type=str, default="data/decomposition/",
                        help="Output directory for decomposition files")
    args = parser.parse_args()

    result = fit_decomposition(
        data_path=args.data_path,
        train_year_max=args.train_year_max,
    )

    # Save
    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, f"decomposition_{args.train_year_max}.json")

    # Convert numpy types for JSON serialization
    serializable = {
        "mu": float(result["mu"]),
        "G": {k: float(v) for k, v in result["G"].items()},
        "E": {k: float(v) for k, v in result["E"].items()},
        "diagnostics": result["diagnostics"],
    }

    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved decomposition to {out_file}")
    print(f"  mu: {result['mu']:.4f}")
    print(f"  G effects: {len(result['G'])} hybrids")
    print(f"  E effects: {len(result['E'])} environments")


if __name__ == "__main__":
    main()

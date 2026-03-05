#!/usr/bin/env python3
"""
Diagnostic script: computes environment fingerprints, target weights,
and per-location statistics for the 2023/2024 validation/test split.

Run from the project root:
    python scripts/diagnose_validation.py

Outputs:
  1. Environment fingerprint distances (2023 ↔ 2024)
  2. Target similarity weights per 2023 location
  3. Sample size distribution (shows Fisher-z reliability)
  4. Comparison of kernel vs density-ratio weighting
  5. Saves weights to data/results/target_weights.json
"""

import sys, os
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import pandas as pd
from utils.target_weighted_validation import (
    compute_environment_fingerprint,
    compute_kernel_similarity_weights,
    compute_density_ratio_weights,
    fisher_z,
)
from utils.validation_integration import (
    _load_numeric_env_features,
    _env_year,
    ENV_CATEGORICAL_COLS,
    TargetWeightedValidator,
)


DATA_DIR = "data/maize_data_2014-2023_vs_2024_v2/"
OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("  VALIDATION TRACKING DIAGNOSTIC")
    print("  2023 (validation) ↔ 2024 (test) environment analysis")
    print("=" * 70)

    # ============================================================
    # 1. Load data
    # ============================================================
    print("\n[1/6] Loading data...")
    df_train = _load_numeric_env_features(DATA_DIR + "X_train.csv")
    df_train["Year"] = df_train["Env"].apply(_env_year)
    df_test = _load_numeric_env_features(DATA_DIR + "X_test.csv")

    numeric_env_cols = [
        c for c in df_train.columns
        if c not in ("Env", "Year") and c not in ENV_CATEGORICAL_COLS
    ]

    val_df = df_train[df_train["Year"] == 2023].copy()
    val_envs = sorted(val_df["Env"].unique())
    test_envs = sorted(df_test["Env"].unique())

    print(f"  Validation (2023): {len(val_envs)} environments, {len(val_df):,} samples")
    print(f"  Test       (2024): {len(test_envs)} environments, {len(df_test):,} samples")

    # Location overlap
    val_sites = {e.rsplit("_", 1)[0] for e in val_envs}
    test_sites = {e.rsplit("_", 1)[0] for e in test_envs}
    overlap = val_sites & test_sites
    only_val = val_sites - test_sites
    only_test = test_sites - val_sites
    print(f"\n  Location overlap: {len(overlap)} shared, "
          f"{len(only_val)} only-in-2023 {sorted(only_val)}, "
          f"{len(only_test)} only-in-2024 {sorted(only_test)}")

    # ============================================================
    # 2. Per-location sample sizes
    # ============================================================
    print("\n[2/6] Sample sizes per location...")
    print(f"\n  {'Env':<20s} {'n_samples':>10s} {'Fisher-z var (1/(n-3))':>22s}")
    print("  " + "-" * 55)
    for env in val_envs:
        n = int((val_df["Env"] == env).sum())
        fz_var = 1.0 / max(n - 3, 1)
        marker = " ← NOISY" if n < 100 else ""
        print(f"  {env:<20s} {n:>10d} {fz_var:>22.6f}{marker}")
    print()
    for env in test_envs:
        n = int((df_test["Env"] == env).sum())
        print(f"  {env:<20s} {n:>10d}  (test)")

    # ============================================================
    # 3. Environment fingerprints
    # ============================================================
    print("\n[3/6] Computing environment fingerprints (mean_std)...")
    val_features = val_df[numeric_env_cols].values.astype(np.float32)
    val_locs = val_df["Env"].values
    val_fps = compute_environment_fingerprint(val_features, val_locs, method="mean_std")

    test_features = df_test[numeric_env_cols].values.astype(np.float32)
    test_locs = df_test["Env"].values
    test_fps = compute_environment_fingerprint(test_features, test_locs, method="mean_std")

    fp_dim = len(next(iter(val_fps.values())))
    print(f"  Fingerprint dimension: {fp_dim}")

    # ============================================================
    # 4. Kernel similarity weights
    # ============================================================
    print("\n[4/6] Computing kernel similarity weights...")
    kernel_weights = compute_kernel_similarity_weights(val_fps, test_fps, tau=None)

    print(f"\n  {'Val Env':<20s} {'Weight':>8s} {'Nearest Test Env':<20s} {'Dist':>8s}")
    print("  " + "-" * 60)

    # Also compute distances for display
    from sklearn.preprocessing import StandardScaler
    all_fp_arr = np.vstack(list(val_fps.values()) + list(test_fps.values()))
    scaler = StandardScaler().fit(all_fp_arr)

    for env in sorted(kernel_weights, key=kernel_weights.get, reverse=True):
        w = kernel_weights[env]
        fp_norm = scaler.transform(val_fps[env].reshape(1, -1))
        best_d, best_t = float("inf"), "?"
        for t_env, t_fp in test_fps.items():
            d = float(np.linalg.norm(scaler.transform(t_fp.reshape(1, -1)) - fp_norm))
            if d < best_d:
                best_d, best_t = d, t_env
        print(f"  {env:<20s} {w:>8.4f} {best_t:<20s} {best_d:>8.3f}")

    # ============================================================
    # 5. Density-ratio weights
    # ============================================================
    print("\n[5/6] Computing density-ratio weights (logistic regression)...")
    dr_weights = compute_density_ratio_weights(val_fps, test_fps)

    print(f"\n  {'Val Env':<20s} {'Kernel w':>10s} {'DensRatio w':>12s} {'Agree?':>8s}")
    print("  " + "-" * 55)
    for env in sorted(val_envs):
        kw = kernel_weights.get(env, 0.0)
        dw = dr_weights.get(env, 0.0)
        # Check if both methods agree on high/low
        kw_rank = sorted(kernel_weights.values(), reverse=True).index(kw) if kw in kernel_weights.values() else -1
        dw_rank = sorted(dr_weights.values(), reverse=True).index(dw) if dw in dr_weights.values() else -1
        agree = "YES" if abs(kw_rank - dw_rank) <= len(val_envs) // 4 else "no"
        print(f"  {env:<20s} {kw:>10.4f} {dw:>12.4f} {agree:>8s}")

    # ============================================================
    # 6. Weight impact analysis
    # ============================================================
    print("\n[6/6] Weight impact analysis...")
    n_per_env = {env: int((val_df["Env"] == env).sum()) for env in val_envs}

    # Unweighted: each location counts equally (leaderboard style)
    # Weighted: W_l = w_target * (n_l - 3)
    print("\n  Effective weight comparison (kernel method):")
    print(f"  {'Env':<20s} {'n':>6s} {'Unweighted':>12s} {'Kernel+Fisher':>14s}")
    print("  " + "-" * 55)

    unw_total = len(val_envs)
    wt_total = sum(kernel_weights.get(e, 1.0) * max(n_per_env[e] - 3, 1) for e in val_envs)

    for env in sorted(val_envs):
        n = n_per_env[env]
        uw = 1.0 / unw_total
        ww = kernel_weights.get(env, 1.0) * max(n - 3, 1) / wt_total
        ratio = ww / uw if uw > 0 else 0
        bar = "█" * int(ratio * 20)
        print(f"  {env:<20s} {n:>6d} {uw:>12.4f} {ww:>14.4f}  {bar}")

    # ============================================================
    # Save weights
    # ============================================================
    print("\n" + "=" * 70)
    print("  Saving target weights...")

    # Save using the validator's built-in method
    validator = TargetWeightedValidator(data_dir=DATA_DIR, verbose=False)
    validator.val_year = 2023
    validator.test_year = 2024
    validator.weighting_method = "kernel"
    validator.target_weights = kernel_weights
    validator.val_envs = np.array(val_envs)
    validator.test_envs = np.array(test_envs)

    weights_path = OUT_DIR / "target_weights.json"
    validator.save_weights(str(weights_path))
    print(f"  Saved to {weights_path}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  • {len(overlap)} of {len(test_sites)} test locations have 2023 counterparts")
    print(f"  • {len(only_test)} test locations are NEW in 2024: {sorted(only_test)}")
    print(f"  • {len(only_val)} 2023 locations absent from 2024: {sorted(only_val)}")
    print(f"    → These get DOWN-weighted by the kernel/density-ratio method")
    print()

    # Find the most and least similar val envs
    sorted_w = sorted(kernel_weights.items(), key=lambda x: x[1], reverse=True)
    print(f"  TOP 5 most representative 2023 envs for 2024:")
    for env, w in sorted_w[:5]:
        print(f"    {env}: weight={w:.4f}")
    print(f"  BOTTOM 5 least representative 2023 envs:")
    for env, w in sorted_w[-5:]:
        print(f"    {env}: weight={w:.4f}")

    print("\n  Next steps:")
    print("  1. Use these weights in train.py for checkpoint selection")
    print("  2. Import create_tw_validator + tw_evaluate from utils/eval_with_target_weighting.py")
    print("  3. Select checkpoints by results['select_score'] instead of raw env_avg_pearson")
    print("=" * 70)


if __name__ == "__main__":
    main()

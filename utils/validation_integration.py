"""
Integration layer: connects target-weighted validation to the existing
GxE_Dataset / train.py pipeline.

Designed to work with the existing data files under
  data/maize_data_2014-2023_vs_2024_v2/

Key mapping from the raw CSVs:
  - Env column  (e.g. "DEH1_2023") encodes *location*_*year*
  - Last 705 columns of X_{train,test}.csv are environment features
  - Categorical env cols (Irrigated, Treatment, Previous_Crop) are the
    first 3 of the 705; the rest are numeric weather/soil covariates.
"""

import json
import warnings
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List

from utils.target_weighted_validation import (
    compute_environment_fingerprint,
    compute_kernel_similarity_weights,
    compute_density_ratio_weights,
    bootstrap_validation_score,
    compute_leaderboard_score,
)


# Number of env feature columns at the tail of X_*.csv
N_ENV = 705
# Columns that are categorical (non-numeric) inside the env block
ENV_CATEGORICAL_COLS = ("Irrigated", "Treatment", "Previous_Crop")


def _env_year(env_str: str) -> int:
    m = re.search(r"(\d{4})$", str(env_str).strip())
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse year from Env='{env_str}'")


def _env_site(env_str: str) -> str:
    """Extract the location / site code from an Env string.
    E.g.  'DEH1_2023' -> 'DEH1',  'ONH3_2024' -> 'ONH3'
    """
    return str(env_str).strip().rsplit("_", 1)[0]


def _load_numeric_env_features(
    csv_path: str,
    envs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Load X_*.csv and return only the *numeric* env feature columns,
    plus the Env column, filtered to `envs` if given.

    Returns a DataFrame with columns: Env + numeric_env_feature_names.
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    all_cols = list(df.columns)

    # Last N_ENV columns are env features
    env_feature_cols = all_cols[-N_ENV:]
    # Keep only numeric ones (drop Irrigated / Treatment / Previous_Crop)
    numeric_env_cols = [c for c in env_feature_cols if c not in ENV_CATEGORICAL_COLS]

    keep = ["Env"] + numeric_env_cols
    sub = df[keep].copy()

    if envs is not None:
        sub = sub[sub["Env"].isin(envs)].reset_index(drop=True)

    # Coerce to float
    for c in numeric_env_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)

    return sub


class TargetWeightedValidator:
    """
    Orchestrates the full target-weighted validation loop.

    Typical usage
    -------------
    >>> validator = TargetWeightedValidator(data_dir="data/maize_data_2014-2023_vs_2024_v2/")
    >>> validator.fit_weights()                # once, at start of training
    >>> results = validator.evaluate(preds, targets, env_ids)
    >>> select_score = results["select_score"] # use for checkpoint selection
    """

    def __init__(
        self,
        data_dir: str = "data/maize_data_2014-2023_vs_2024_v2/",
        val_year: int = 2023,
        test_year: int = 2024,
        weighting_method: str = "kernel",      # "kernel" | "density_ratio"
        fingerprint_method: str = "mean_std",   # "mean" | "mean_std" | "quantiles"
        tau: Optional[float] = None,
        n_bootstrap: int = 1000,
        pessimistic_quantile: float = 0.10,
        min_samples: int = 5,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.val_year = val_year
        self.test_year = test_year
        self.weighting_method = weighting_method
        self.fingerprint_method = fingerprint_method
        self.tau = tau
        self.n_bootstrap = n_bootstrap
        self.pessimistic_quantile = pessimistic_quantile
        self.min_samples = min_samples
        self.verbose = verbose

        # populated by fit_weights()
        self.target_weights: Optional[Dict[str, float]] = None
        self.val_fingerprints: Optional[Dict[str, np.ndarray]] = None
        self.test_fingerprints: Optional[Dict[str, np.ndarray]] = None
        self.numeric_env_cols: Optional[List[str]] = None
        self.val_envs: Optional[np.ndarray] = None
        self.test_envs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit_weights(self):
        """
        Load train + test CSVs, compute environment fingerprints for
        the validation year and test year, then derive target weights.

        Only reads the Env column + 702 numeric env features — fast.
        """
        x_train_path = str(self.data_dir / "X_train.csv")
        x_test_path = str(self.data_dir / "X_test.csv")

        # ---- validation year data (from training file) ----
        df_train_env = _load_numeric_env_features(x_train_path)
        df_train_env["Year"] = df_train_env["Env"].apply(_env_year)
        val_df = df_train_env[df_train_env["Year"] == self.val_year].copy()
        self.val_envs = val_df["Env"].unique()

        # ---- test year data ----
        df_test_env = _load_numeric_env_features(x_test_path)
        self.test_envs = df_test_env["Env"].unique()

        # Identify numeric feature columns (same in both)
        self.numeric_env_cols = [
            c for c in val_df.columns
            if c not in ("Env", "Year") and c not in ENV_CATEGORICAL_COLS
        ]

        # ---- fingerprints ----
        val_features = val_df[self.numeric_env_cols].values
        val_locs = val_df["Env"].values
        self.val_fingerprints = compute_environment_fingerprint(
            val_features, val_locs, method=self.fingerprint_method,
        )

        test_features = df_test_env[self.numeric_env_cols].values
        test_locs = df_test_env["Env"].values
        self.test_fingerprints = compute_environment_fingerprint(
            test_features, test_locs, method=self.fingerprint_method,
        )

        # ---- weights ----
        if self.weighting_method == "kernel":
            self.target_weights = compute_kernel_similarity_weights(
                self.val_fingerprints, self.test_fingerprints, tau=self.tau,
            )
        elif self.weighting_method == "density_ratio":
            self.target_weights = compute_density_ratio_weights(
                self.val_fingerprints, self.test_fingerprints,
            )
        else:
            raise ValueError(f"Unknown weighting_method: {self.weighting_method}")

        if self.verbose:
            self._print_weight_summary()

        return self

    # ------------------------------------------------------------------
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        env_ids: np.ndarray,
    ) -> Dict:
        """
        Run the full target-weighted, bootstrapped evaluation.

        Parameters
        ----------
        predictions, targets : np.ndarray shape (N,)
        env_ids : np.ndarray shape (N,)  — Env strings (e.g. "DEH1_2023")

        Returns
        -------
        dict with:
            select_score   — USE THIS for checkpoint / hparam selection
            point_estimate — single-pass weighted score
            leaderboard_style — standard unweighted macro-avg r
            mean, std, pessimistic, optimistic, median
            per_location_r — raw per-location correlations
        """
        if self.target_weights is None:
            raise RuntimeError("Call fit_weights() before evaluate()!")

        # Ensure string env ids
        env_ids_str = np.array([str(e) for e in env_ids])

        boot = bootstrap_validation_score(
            predictions, targets, env_ids_str,
            target_weights=self.target_weights,
            n_bootstrap=self.n_bootstrap,
            pessimistic_quantile=self.pessimistic_quantile,
            use_fisher_z=True,
            weight_by_n=True,
            min_samples=self.min_samples,
        )

        lb_score, lb_per_loc = compute_leaderboard_score(
            predictions, targets, env_ids_str, min_samples=2,
        )

        results = {
            **boot,
            "leaderboard_style": lb_score,
            "leaderboard_per_location": lb_per_loc,
        }
        return results

    # ------------------------------------------------------------------
    def print_evaluation(self, results: Dict, prefix: str = ""):
        print(f"\n{'='*60}")
        print(f"{prefix}Validation Results")
        print(f"{'='*60}")
        print(f"  SELECT SCORE (pessimistic q={self.pessimistic_quantile:.0%}): "
              f"{results['select_score']:.4f}")
        print(f"  Point estimate (target-weighted):  {results['point_estimate']:.4f}")
        print(f"  Leaderboard-style (unweighted):    {results['leaderboard_style']:.4f}")
        print(f"  Bootstrap mean ± std:              {results['mean']:.4f} ± {results['std']:.4f}")
        print(f"  Bootstrap [{self.pessimistic_quantile:.0%}, {1-self.pessimistic_quantile:.0%}]: "
              f"[{results['pessimistic']:.4f}, {results['optimistic']:.4f}]")
        print()
        print(f"  Per-location raw Pearson r:")
        plr = results.get("per_location_r", {})
        for loc in sorted(plr.keys()):
            r = plr[loc]
            w = self.target_weights.get(loc, 0.0)
            print(f"    {loc:20s}  r={r:+.4f}  weight={w:.4f}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    def save_weights(self, path: str):
        data = {
            "val_year": self.val_year,
            "test_year": self.test_year,
            "weighting_method": self.weighting_method,
            "fingerprint_method": self.fingerprint_method,
            "target_weights": {str(k): float(v) for k, v in self.target_weights.items()},
            "val_envs": [str(e) for e in (self.val_envs if self.val_envs is not None else [])],
            "test_envs": [str(e) for e in (self.test_envs if self.test_envs is not None else [])],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_weights(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.target_weights = {k: float(v) for k, v in data["target_weights"].items()}
        self.val_year = data.get("val_year", self.val_year)
        self.test_year = data.get("test_year", self.test_year)
        self.weighting_method = data.get("weighting_method", self.weighting_method)
        return self

    # ------------------------------------------------------------------
    def _print_weight_summary(self):
        w = self.target_weights
        print(f"\n{'='*60}")
        print(f"Target-Weighted Validation Setup")
        print(f"{'='*60}")
        print(f"  Val year:  {self.val_year}  ({len(self.val_fingerprints)} envs)")
        print(f"  Test year: {self.test_year} ({len(self.test_fingerprints)} envs)")
        print(f"  Method:    {self.weighting_method}")
        print(f"  Weight range: [{min(w.values()):.4f}, {max(w.values()):.4f}]")
        print()
        # Sort by weight so most-representative validation envs are at top
        for loc, wt in sorted(w.items(), key=lambda x: -x[1]):
            # Find nearest test env for interpretability
            if self.test_fingerprints:
                from sklearn.preprocessing import StandardScaler as SS2
                all_fps = np.vstack(list(self.val_fingerprints.values()) + list(self.test_fingerprints.values()))
                sc = SS2().fit(all_fps)
                fp_norm = sc.transform(self.val_fingerprints[loc].reshape(1, -1))
                best_d, best_test_loc = float("inf"), "?"
                for tloc, tfp in self.test_fingerprints.items():
                    d = np.linalg.norm(sc.transform(tfp.reshape(1, -1)) - fp_norm)
                    if d < best_d:
                        best_d, best_test_loc = d, tloc
                print(f"  {loc:20s}  w={wt:.4f}  (nearest test: {best_test_loc}, d={best_d:.3f})")
            else:
                print(f"  {loc:20s}  w={wt:.4f}")
        print(f"{'='*60}\n")

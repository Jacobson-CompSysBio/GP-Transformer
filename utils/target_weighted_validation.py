"""
Target-weighted, variance-stabilized validation scheme for GxE maize prediction.

Addresses the train/validation/test tracking problem by:
1. Reweighting 2023 validation locations to match 2024 test environment distribution
2. Fisher-z stabilizing per-location correlations
3. Bootstrap-based pessimistic model selection

Reference: Sugiyama et al. (JMLR 2007) - importance-weighted CV under covariate shift

The 705 env features in the data are:
  - latitude, longitude
  - 3 categorical (Irrigated, Treatment, Previous_Crop) — dropped or one-hot encoded
  - ~700 numeric weather/soil features (RH2M_min, T2M_MAX_min, ..., LL__1-LL__10)

Location (site) is the prefix of the Env column (e.g., "DEH1" from "DEH1_2023").
"""

import warnings
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional, Set


# ---------------------------------------------------------------------------
# Step 1: Environment fingerprinting
# ---------------------------------------------------------------------------

def compute_environment_fingerprint(
    env_features: np.ndarray,
    location_ids: np.ndarray,
    method: str = "mean_std",
) -> Dict[str, np.ndarray]:
    """
    Compute a fixed-length environment fingerprint per location.

    Parameters
    ----------
    env_features : np.ndarray, shape (n_samples, n_env_features)
        Numeric environment covariates (already scaled or raw — will be
        standardised internally).
    location_ids : np.ndarray, shape (n_samples,)
        Location identifier per sample (e.g. "DEH1_2023").
    method : str
        "mean"       – mean of env features per location
        "mean_std"   – concat(mean, std) per location
        "quantiles"  – mean, std, q10, q50, q90

    Returns
    -------
    fingerprints : dict  {location_id: np.ndarray}
    """
    unique_locs = np.unique(location_ids)
    fingerprints: Dict[str, np.ndarray] = {}

    for loc in unique_locs:
        loc_mask = location_ids == loc
        loc_env = env_features[loc_mask].astype(np.float64)

        if method == "mean":
            fp = np.mean(loc_env, axis=0)
        elif method == "mean_std":
            fp = np.concatenate([
                np.mean(loc_env, axis=0),
                np.std(loc_env, axis=0, ddof=0),
            ])
        elif method == "quantiles":
            fp = np.concatenate([
                np.mean(loc_env, axis=0),
                np.std(loc_env, axis=0, ddof=0),
                np.percentile(loc_env, 10, axis=0),
                np.percentile(loc_env, 50, axis=0),
                np.percentile(loc_env, 90, axis=0),
            ])
        else:
            raise ValueError(f"Unknown fingerprint method: {method}")

        fingerprints[str(loc)] = fp

    return fingerprints


# ---------------------------------------------------------------------------
# Step 2: Target similarity weights
# ---------------------------------------------------------------------------

def compute_kernel_similarity_weights(
    val_fingerprints: Dict[str, np.ndarray],
    test_fingerprints: Dict[str, np.ndarray],
    tau: Optional[float] = None,
) -> Dict[str, float]:
    """
    Kernel-similarity weighting (Option A).

    For each validation location, weight = exp(-d²/τ) where d is the
    minimum L2 distance to any test location fingerprint (after
    standardisation).

    Parameters
    ----------
    val_fingerprints, test_fingerprints : dict {loc: np.ndarray}
    tau : float or None
        Bandwidth.  If None, use median-heuristic (median of d²).

    Returns
    -------
    weights : dict {val_location: float}
    """
    test_fps = np.array(list(test_fingerprints.values()))

    # Standardise jointly
    all_fps = np.vstack([
        np.array(list(val_fingerprints.values())),
        test_fps,
    ])
    scaler = StandardScaler()
    scaler.fit(all_fps)
    test_fps_norm = scaler.transform(test_fps)

    distances: Dict[str, float] = {}
    for loc, fp in val_fingerprints.items():
        fp_norm = scaler.transform(fp.reshape(1, -1))
        dists = np.linalg.norm(test_fps_norm - fp_norm, axis=1)
        distances[loc] = float(np.min(dists))

    # Bandwidth: median heuristic
    all_d2 = np.array([d ** 2 for d in distances.values()])
    if tau is None:
        tau = float(np.median(all_d2))
        if tau < 1e-10:
            tau = 1.0

    weights = {loc: float(np.exp(-(d ** 2) / tau))
               for loc, d in distances.items()}
    return weights


def compute_density_ratio_weights(
    val_fingerprints: Dict[str, np.ndarray],
    test_fingerprints: Dict[str, np.ndarray],
    clip_max: float = 10.0,
) -> Dict[str, float]:
    """
    Density-ratio weighting via logistic regression (Option B).

    Fit a classifier to distinguish test vs validation location
    fingerprints, then derive w(x) ≈ p_test(x) / p_val(x).

    Parameters
    ----------
    val_fingerprints, test_fingerprints : dict {loc: np.ndarray}
    clip_max : float  – max weight to prevent instability

    Returns
    -------
    weights : dict {val_location: float}
    """
    val_locs = list(val_fingerprints.keys())
    test_locs = list(test_fingerprints.keys())

    val_X = np.array([val_fingerprints[loc] for loc in val_locs])
    test_X = np.array([test_fingerprints[loc] for loc in test_locs])

    scaler = StandardScaler()
    X_all = np.vstack([val_X, test_X])
    scaler.fit(X_all)
    val_X_norm = scaler.transform(val_X)
    test_X_norm = scaler.transform(test_X)

    if len(val_X_norm) < 3 or len(test_X_norm) < 3:
        warnings.warn("Too few locations for density-ratio estimation. Returning uniform weights.")
        return {loc: 1.0 for loc in val_locs}

    X = np.vstack([val_X_norm, test_X_norm])
    y = np.concatenate([np.zeros(len(val_X_norm)), np.ones(len(test_X_norm))])

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)

    val_probs = clf.predict_proba(val_X_norm)[:, 1]
    prevalence_correction = len(val_X_norm) / len(test_X_norm)

    weights: Dict[str, float] = {}
    for i, loc in enumerate(val_locs):
        p = float(np.clip(val_probs[i], 0.01, 0.99))
        ratio = (p / (1.0 - p)) * prevalence_correction
        weights[loc] = float(np.clip(ratio, 0.01, clip_max))

    return weights


# ---------------------------------------------------------------------------
# Step 3: Fisher-z stabilised, target-weighted aggregation
# ---------------------------------------------------------------------------

def fisher_z(r: float) -> float:
    """Fisher z-transform: z = atanh(r). Variance ≈ 1/(n-3)."""
    return float(np.arctanh(np.clip(r, -0.999, 0.999)))


def inv_fisher_z(z: float) -> float:
    """Inverse Fisher z: r = tanh(z)."""
    return float(np.tanh(z))


def compute_target_weighted_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    location_ids: np.ndarray,
    target_weights: Dict[str, float],
    use_fisher_z: bool = True,
    weight_by_n: bool = True,
    min_samples: int = 5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute target-weighted, Fisher-z-stabilised validation score.

    Parameters
    ----------
    predictions, targets : np.ndarray, shape (N,)
    location_ids : np.ndarray, shape (N,)
    target_weights : dict {location: float}
    use_fisher_z : bool – apply Fisher z variance stabilisation
    weight_by_n : bool – additionally weight by (n_l - 3)
    min_samples : int – skip locations with fewer samples

    Returns
    -------
    score : float – aggregated score (back-transformed to correlation scale)
    per_location : dict {location: raw pearson r}
    """
    unique_locs = np.unique(location_ids)
    per_location: Dict[str, float] = {}
    z_values: Dict[str, float] = {}
    combined_weights: Dict[str, float] = {}

    for loc in unique_locs:
        loc_str = str(loc)
        mask = location_ids == loc
        n = int(mask.sum())

        if n < min_samples:
            continue
        pred_loc = predictions[mask]
        targ_loc = targets[mask]
        if np.std(pred_loc) < 1e-10 or np.std(targ_loc) < 1e-10:
            continue

        r, _ = stats.pearsonr(pred_loc, targ_loc)
        per_location[loc_str] = float(r)

        # Only include locations that have explicit target weights.
        # Locations absent from the weight dict (e.g. non-2023 envs when
        # weights were derived for 2023→2024) are excluded entirely.
        w_target = target_weights.get(loc_str, 0.0)
        if w_target <= 0.0:
            continue

        z = fisher_z(r) if use_fisher_z else r
        z_values[loc_str] = z

        w_n = max(n - 3, 1) if weight_by_n else 1.0
        combined_weights[loc_str] = w_target * w_n

    if not z_values:
        warnings.warn("No valid locations for scoring!")
        return 0.0, per_location

    locs = list(z_values.keys())
    z_arr = np.array([z_values[l] for l in locs])
    w_arr = np.array([combined_weights[l] for l in locs])
    w_arr = w_arr / w_arr.sum()

    z_bar = float(np.sum(w_arr * z_arr))
    score = inv_fisher_z(z_bar) if use_fisher_z else z_bar
    return score, per_location


# ---------------------------------------------------------------------------
# Step 4: Bootstrap pessimistic selection
# ---------------------------------------------------------------------------

def bootstrap_validation_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    location_ids: np.ndarray,
    target_weights: Dict[str, float],
    n_bootstrap: int = 1000,
    pessimistic_quantile: float = 0.10,
    seed: int = 42,
    use_fisher_z: bool = True,
    weight_by_n: bool = True,
    min_samples: int = 5,
) -> Dict[str, object]:
    """
    Bootstrap over locations to get robust model-selection statistics.

    Returns
    -------
    dict with keys:
        mean, std, pessimistic, optimistic, median,
        point_estimate, select_score, per_location_r
    """
    rng = np.random.RandomState(seed)

    point_score, per_location = compute_target_weighted_score(
        predictions, targets, location_ids, target_weights,
        use_fisher_z=use_fisher_z, weight_by_n=weight_by_n,
        min_samples=min_samples,
    )

    valid_locs = np.array(list(per_location.keys()))
    if len(valid_locs) < 3:
        warnings.warn("Too few valid locations for bootstrap!")
        return {
            "mean": point_score, "std": 0.0,
            "pessimistic": point_score, "optimistic": point_score,
            "median": point_score, "point_estimate": point_score,
            "select_score": point_score,
            "per_location_r": per_location,
        }

    boot_scores = []
    for _ in range(n_bootstrap):
        boot_locs = rng.choice(valid_locs, size=len(valid_locs), replace=True)
        boot_preds, boot_targs, boot_loc_ids = [], [], []
        for loc in boot_locs:
            mask = location_ids == loc
            boot_preds.append(predictions[mask])
            boot_targs.append(targets[mask])
            boot_loc_ids.append(np.full(mask.sum(), loc))
        bp = np.concatenate(boot_preds)
        bt = np.concatenate(boot_targs)
        bl = np.concatenate(boot_loc_ids)
        s, _ = compute_target_weighted_score(
            bp, bt, bl, target_weights,
            use_fisher_z=use_fisher_z, weight_by_n=weight_by_n,
            min_samples=min_samples,
        )
        boot_scores.append(s)

    boot_scores = np.array(boot_scores)
    return {
        "mean": float(np.mean(boot_scores)),
        "std": float(np.std(boot_scores)),
        "pessimistic": float(np.percentile(boot_scores, pessimistic_quantile * 100)),
        "optimistic": float(np.percentile(boot_scores, (1 - pessimistic_quantile) * 100)),
        "median": float(np.median(boot_scores)),
        "point_estimate": float(point_score),
        "select_score": float(np.percentile(boot_scores, pessimistic_quantile * 100)),
        "per_location_r": per_location,
    }


# ---------------------------------------------------------------------------
# Baseline: plain leaderboard-style metric (for comparison)
# ---------------------------------------------------------------------------

def compute_leaderboard_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    location_ids: np.ndarray,
    min_samples: int = 2,
) -> Tuple[float, Dict[str, float]]:
    """Unweighted macro-avg per-location Pearson r (competition metric)."""
    unique_locs = np.unique(location_ids)
    per_loc: Dict[str, float] = {}
    for loc in unique_locs:
        loc_str = str(loc)
        mask = location_ids == loc
        if mask.sum() < min_samples:
            continue
        p, t = predictions[mask], targets[mask]
        if np.std(p) < 1e-10 or np.std(t) < 1e-10:
            continue
        r, _ = stats.pearsonr(p, t)
        per_loc[loc_str] = float(r)
    if not per_loc:
        return 0.0, per_loc
    return float(np.mean(list(per_loc.values()))), per_loc


# ---------------------------------------------------------------------------
# Genotype fingerprinting (for genotype covariate shift correction)
# ---------------------------------------------------------------------------

def _parse_hybrid_parts(sample_id: str) -> Tuple[str, str, str]:
    """Parse 'ENV-parent1/parent2' → (hybrid_str, parent1, parent2)."""
    parts = str(sample_id).split("-", 1)
    if len(parts) < 2:
        return str(sample_id), "", ""
    hybrid_str = parts[1]
    pp = hybrid_str.split("/", 1)
    return hybrid_str, pp[0], pp[1] if len(pp) > 1 else ""


def compute_genotype_fingerprint(
    sample_ids: np.ndarray,
    env_ids: np.ndarray,
    prior_hybrids: Optional[Set[str]] = None,
    all_testers: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Per-environment genotype fingerprint: tester distribution + novelty rate.

    For each environment, produces a vector:
      [frac_tester_0, frac_tester_1, ..., frac_tester_K, novelty_fraction]

    Parameters
    ----------
    sample_ids : array of str
        Sample IDs in format "{Env}-{parent1}/{parent2}".
    env_ids : array of str
        Environment identifier per sample (e.g. "DEH1_2023").
    prior_hybrids : set of str or None
        Hybrid IDs from prior training years, used to compute novelty
        fraction.  If None, novelty is set to 0.
    all_testers : list of str or None
        Ordered tester vocabulary.  If None, discovered from data.

    Returns
    -------
    fingerprints : dict {env_id: np.ndarray}
    all_testers : list of str — the tester ordering used
    """
    parsed = [_parse_hybrid_parts(sid) for sid in sample_ids]
    hybrids = np.array([p[0] for p in parsed])
    parent2s = np.array([p[2] for p in parsed])

    if all_testers is None:
        all_testers = sorted(set(parent2s) - {""})
    tester_idx = {t: i for i, t in enumerate(all_testers)}
    n_testers = len(all_testers)

    fingerprints: Dict[str, np.ndarray] = {}
    for env in np.unique(env_ids):
        mask = env_ids == env
        env_p2 = parent2s[mask]
        env_hy = hybrids[mask]
        n = int(mask.sum())

        # Tester distribution histogram (normalized)
        hist = np.zeros(n_testers, dtype=np.float64)
        for p2 in env_p2:
            idx = tester_idx.get(p2)
            if idx is not None:
                hist[idx] += 1
        if n > 0:
            hist /= n

        # Novelty fraction
        novelty = 0.0
        if prior_hybrids is not None:
            uniq = set(env_hy)
            novelty = sum(1 for h in uniq if h not in prior_hybrids) / max(len(uniq), 1)

        fingerprints[str(env)] = np.concatenate([hist, [novelty]])

    return fingerprints, all_testers


def compute_genotype_diversity_weights(
    val_fingerprints: Dict[str, np.ndarray],
    test_fingerprints: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Genotype weights based on tester diversity (Shannon entropy).

    Upweights validation environments with more diverse tester
    compositions, which test the model's cross-tester generalization
    ability — more informative about test performance with novel testers.

    The RBF kernel approach (compute_kernel_similarity_weights) can
    invert the desired signal when val and test tester distributions
    are extremely different: it upweights val envs with the LEAST
    minor-tester diversity (since those are numerically closest to
    test envs that also have trace amounts of minor testers).

    Parameters
    ----------
    val_fingerprints : dict {env_id: np.ndarray}
        From compute_genotype_fingerprint.  Last element is novelty.
    test_fingerprints : dict {env_id: np.ndarray}
        Used only to compute the target entropy for normalization.

    Returns
    -------
    weights : dict {val_env: float}
    """
    # Compute entropy for each val env (tester histogram only, not novelty)
    entropies: Dict[str, float] = {}
    for env, fp in val_fingerprints.items():
        tester_hist = fp[:-1]  # exclude novelty
        nonzero = tester_hist[tester_hist > 0]
        if len(nonzero) > 0:
            entropies[env] = float(-np.sum(nonzero * np.log(nonzero)))
        else:
            entropies[env] = 0.0

    # Normalize to [0, 1] using max observed entropy
    max_ent = max(entropies.values()) if entropies else 1.0
    if max_ent < 1e-10:
        return {env: 1.0 for env in val_fingerprints}

    weights = {env: h / max_ent for env, h in entropies.items()}
    return weights


def combine_weights(
    env_weights: Dict[str, float],
    geno_weights: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Combine environment and genotype similarity weights multiplicatively.

    w_final = w_env^(1 - alpha) * w_geno^alpha

    alpha=0.0 → pure environment weights (backward compatible)
    alpha=1.0 → pure genotype weights
    """
    all_locs = set(env_weights) | set(geno_weights)
    combined: Dict[str, float] = {}
    for loc in all_locs:
        we = max(env_weights.get(loc, 0.0), 0.0)
        wg = max(geno_weights.get(loc, 0.0), 0.0)
        if we <= 0 and alpha < 1.0:
            combined[loc] = 0.0
        elif wg <= 0 and alpha > 0.0:
            combined[loc] = 0.0
        else:
            combined[loc] = float((we ** (1.0 - alpha)) * (wg ** alpha))
    return combined

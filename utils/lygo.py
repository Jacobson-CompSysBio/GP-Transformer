"""
utils/lygo.py — Phase 0 validation: LYGO (Leave-Year-and-Genotype-Out)

Why this exists
---------------
`analyze_val_failure.py` / `analyze_holdout_strategies.py` showed that NO current
scheme tracks the 2024 test:

    2024 test  = NOVEL parent1  x  KNOWN tester (PHP02/LH287)  x  NOVEL year
    LEO        = known hybrids  x  held-out location           (spatial only)
    2023 year  = known hybrids  x  novel weather               (temporal only)
    proxy_same_tester = NOVEL parent1 x known tester           (genetic only; same years)

LYGO is the only split that reproduces BOTH novelty axes at once. It is the
`proxy_same_tester` scheme (novel parent1 -> known tester) restricted to a held-out
*year*, with training drawn strictly from earlier years:

    train = all rows with Year <  val_year
    val   = rows with Year == val_year  whose parent1 is NOVEL (unseen in Year<val_year)
            and whose tester (parent2) is in `testers`
            (and whose env has >= min_val_per_env novel-cross rows)

Because parent1 in `val` never appears before `val_year`, train (Year<val_year)
cannot leak it -> genetic novelty holds by construction. The held-out year gives
temporal novelty. Known-parent1 / non-target-tester rows in `val_year` are simply
*excluded* from both splits (they don't match the 2024 structure).

TWO-STAGE USAGE (important)
---------------------------
LYGO is a *selector*, not the final-model trainer. It deliberately trains on less
data (Year < val_year) so the validation number is comparable to the 2024 task.

    1. SELECT: run LYGO (ideally several folds x seeds, see run_lygo_folds) to RANK
       configs by mean +/- std val env-PCC.
    2. SUBMIT: retrain the winning config on ALL pre-2024 data (the normal trunk),
       then evaluate on the real 2024 test split (X_test). Do NOT submit the
       Year<val_year selector model.

Drop-in: either keep this as `utils/lygo.py` and import the three core helpers into
`utils/dataset.py`, or paste them next to `compute_leo_val_envs` /
`compute_proxy_same_tester_holdout`. The `__init__` branch, CLI args, run-name tag,
and train.py wiring to add are in the clearly-marked comment blocks near the bottom.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Tiny parsers. These already exist in utils/dataset.py — included here only so
# this module is standalone. DELETE these three if you paste the helpers into
# dataset.py (use the module-level versions there instead).
# ----------------------------------------------------------------------------
def _env_year_from_str(env_str: str) -> int:
    m = re.search(r"(\d{4})$", str(env_str).strip())
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse year from Env='{env_str}'")


def extract_hybrid_name(sample_id: str) -> str:
    sample_id = str(sample_id)
    return sample_id.split("-", 1)[1] if "-" in sample_id else sample_id


def split_hybrid_parents(hybrid: pd.Series) -> Tuple[pd.Series, pd.Series]:
    parts = hybrid.astype(str).str.split("/", n=1, expand=True)
    parent1 = parts[0].fillna("").astype(str).str.strip()
    if parts.shape[1] > 1:
        parent2 = parts[1].fillna("").astype(str).str.strip()
    else:
        parent2 = pd.Series("", index=hybrid.index)
    return parent1, parent2


def _ensure_meta_cols(x_raw: pd.DataFrame) -> pd.DataFrame:
    """Guarantee Year/Hybrid/parent1/parent2 exist. No-op if already derived
    (as they are inside GxE_Dataset.__init__ before the split logic runs)."""
    x = x_raw
    need = [c for c in ("Year", "Hybrid", "parent1", "parent2") if c not in x.columns]
    if not need:
        return x
    x = x.copy()
    if "Year" not in x.columns:
        x["Year"] = x["Env"].astype(str).map(_env_year_from_str)
    if "Hybrid" not in x.columns:
        x["Hybrid"] = x["id"].astype(str).map(extract_hybrid_name)
    if "parent1" not in x.columns or "parent2" not in x.columns:
        p1, p2 = split_hybrid_parents(x["Hybrid"])
        x["parent1"], x["parent2"] = p1, p2
    return x


# ----------------------------------------------------------------------------
# Core helper 1: derive the tester set straight from the 2024 test file, so the
# selector faces the exact testers the model will be scored against.
# ----------------------------------------------------------------------------
def compute_test_testers(data_path: str, test_year: int = 2024) -> set[str]:
    """Set of parent2 (tester) lines present in the test_year split of X_test.csv.

    Falls back to an empty set (caller should then pass `testers` explicitly) if
    X_test is unavailable.
    """
    try:
        xt = pd.read_csv(
            data_path + "X_test.csv",
            usecols=lambda c: c in ("id", "Env"),
        )
    except (FileNotFoundError, ValueError):
        return set()
    yr = xt["Env"].astype(str).map(_env_year_from_str)
    hyb = xt["id"].astype(str).map(extract_hybrid_name)
    _, p2 = split_hybrid_parents(hyb)
    return set(p2[yr == test_year].unique()) - {""}


# ----------------------------------------------------------------------------
# Core helper 2: the boolean masks. Deterministic given (val_year, testers,
# novelty, valid_val_envs) and a fixed X_train, so train and val calls agree
# even if recomputed independently.
# ----------------------------------------------------------------------------
def lygo_split_masks(
    x_raw: pd.DataFrame,
    val_year: int,
    testers: Iterable[str],
    novelty: str,
    valid_val_envs: Iterable[str],
) -> Tuple[pd.Series, pd.Series]:
    """Return (train_mask, val_mask) boolean Series aligned to x_raw.

    Handles both novelty modes correctly:
      - "parent1": val parent1 unseen in any earlier year (combining-ability transfer)
      - "hybrid" : val hybrid  unseen in any earlier year (novel cross, looser)
    """
    x = _ensure_meta_cols(x_raw)
    testers = set(testers)
    valid_val_envs = set(valid_val_envs)

    earlier = x["Year"] < val_year
    in_year = x["Year"] == val_year
    target = x["parent2"].isin(testers)

    if novelty == "parent1":
        seen = set(x.loc[earlier, "parent1"].unique())
        novel = ~x["parent1"].isin(seen)
    elif novelty == "hybrid":
        seen = set(x.loc[earlier, "Hybrid"].unique())
        novel = ~x["Hybrid"].isin(seen)
    else:
        raise ValueError(f"novelty must be 'parent1' or 'hybrid' (got {novelty!r})")

    val_mask = in_year & target & novel & x["Env"].isin(valid_val_envs)
    train_mask = earlier
    return train_mask, val_mask


# ----------------------------------------------------------------------------
# Core helper 3: resolve the held-out envs (apply the per-env min-count filter)
# and emit diagnostics. Call this ONCE on the train split; pass `valid_val_envs`
# (and the resolved year/testers) to the val split for consistency.
# ----------------------------------------------------------------------------
def compute_lygo_holdout(
    x_raw: pd.DataFrame,
    val_year: int,
    testers: Iterable[str],
    novelty: str = "parent1",
    min_val_per_env: int = 4,
    test_testers: Optional[Iterable[str]] = None,
) -> Tuple[set[str], Dict[str, object]]:
    """Returns (valid_val_envs, info).

    valid_val_envs: envs in `val_year` that contain >= min_val_per_env novel-cross
    rows (envs with too few are dropped so per-env PCC is estimable).
    info: diagnostics, including hard warnings if the val set is too thin to give a
    stable selector signal.
    """
    x = _ensure_meta_cols(x_raw)
    testers = set(testers)

    earlier = x[x["Year"] < val_year]
    in_year = x[x["Year"] == val_year]
    if earlier.empty:
        raise ValueError(f"No training rows with Year < {val_year}.")
    if in_year.empty:
        raise ValueError(f"No rows in val_year={val_year}.")

    seen_p1 = set(earlier["parent1"].unique())
    seen_hyb = set(earlier["Hybrid"].unique())

    cand = in_year[in_year["parent2"].isin(testers)]
    if novelty == "parent1":
        novel = cand[~cand["parent1"].isin(seen_p1)]
    elif novelty == "hybrid":
        novel = cand[~cand["Hybrid"].isin(seen_hyb)]
    else:
        raise ValueError(f"novelty must be 'parent1' or 'hybrid' (got {novelty!r})")

    env_counts_all = novel["Env"].value_counts()
    if min_val_per_env and min_val_per_env > 1:
        valid_val_envs = set(env_counts_all[env_counts_all >= min_val_per_env].index)
    else:
        valid_val_envs = set(env_counts_all.index)

    kept = novel[novel["Env"].isin(valid_val_envs)]
    env_counts_kept = kept["Env"].value_counts()
    val_parent1s = set(kept["parent1"].unique())
    val_hybrids = set(kept["Hybrid"].unique())

    warnings: list[str] = []
    if len(kept) < 200:
        warnings.append(
            f"Only {len(kept)} val rows — env-PCC will be noisy; consider a year "
            f"with more novel crosses, novelty='hybrid', or broader testers."
        )
    if len(valid_val_envs) < 5:
        warnings.append(
            f"Only {len(valid_val_envs)} val envs survive the min_val_per_env="
            f"{min_val_per_env} filter — macro env-PCC averages over too few envs."
        )
    overlap = val_parent1s & seen_p1
    if novelty == "parent1" and overlap:
        warnings.append(
            f"{len(overlap)} val parent1s also appear before {val_year} — novelty "
            f"violated (check parse). Example: {sorted(overlap)[:3]}"
        )
    if not testers:
        warnings.append("Empty tester set — pass `testers` or ensure X_test exists.")

    info: Dict[str, object] = {
        "scheme": "lygo",
        "val_year": int(val_year),
        "novelty": novelty,
        "testers": sorted(testers),
        "min_val_per_env": int(min_val_per_env),
        "n_val_rows": int(len(kept)),
        "n_val_envs": int(len(valid_val_envs)),
        "n_val_parent1s": int(len(val_parent1s)),
        "n_val_hybrids": int(len(val_hybrids)),
        "val_rows_per_env_min": int(env_counts_kept.min()) if len(env_counts_kept) else 0,
        "val_rows_per_env_median": float(env_counts_kept.median()) if len(env_counts_kept) else 0.0,
        "n_envs_dropped_by_filter": int(len(env_counts_all) - len(valid_val_envs)),
        "n_envs_below_4_before_filter": int((env_counts_all < 4).sum()),
        "train_years": [int(y) for y in sorted(earlier["Year"].unique())],
        "train_rows": int(len(earlier)),
        "train_parent1s": int(len(seen_p1)),
        "overlap_val_train_parent1s": int(len(overlap)),
        "warnings": warnings,
    }
    if test_testers is not None:
        tt = set(test_testers)
        present = set(in_year["parent2"].unique())
        info["test_testers"] = sorted(tt)
        info["test_testers_missing_in_val_year"] = sorted(tt - present)
    return valid_val_envs, info


# ----------------------------------------------------------------------------
# Did the validator actually validate? Empirical check that LYGO ranks configs
# more like the 2024 test than LEO does. Feed it a table of PAST runs once you
# have a handful; this is the Phase 0 "bar".
# ----------------------------------------------------------------------------
def selector_rank_tracking(rows: list[dict]) -> Dict[str, float]:
    """rows: [{'config': str, 'lygo_pcc': float, 'leo_pcc': float,
              'test_pcc': float}, ...]  (>= 4 runs recommended)

    Returns Spearman rank-correlation of each selector vs the real 2024 test PCC.
    Higher = the selector orders configs more like the test does. LYGO should beat
    LEO here; if it doesn't, fall back to ensemble-over-configs (Phase 5).
    """
    from scipy.stats import spearmanr

    test = np.asarray([r["test_pcc"] for r in rows], dtype=float)
    out: Dict[str, float] = {}
    for key in ("lygo_pcc", "leo_pcc"):
        vals = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = float(spearmanr(vals, test).correlation)
    out["n_runs"] = float(len(rows))
    out["lygo_beats_leo"] = float(out["lygo_pcc"] > out["leo_pcc"])
    return out


# ----------------------------------------------------------------------------
# Fold/seed harness. Folds = which year you hold out; seeds = model init. A
# single LYGO year is one noisy estimate; averaging over a few recent years and
# >=3 seeds gives the mean +/- std band the decision protocol (AGENTS.md §1)
# requires. Plug your training entrypoint into `train_and_eval`.
# ----------------------------------------------------------------------------
def run_lygo_folds(
    train_and_eval,
    val_years: Iterable[int] = (2021, 2022, 2023),
    seeds: Iterable[int] = (1, 2, 3),
) -> Tuple[Dict[str, object], list[dict]]:
    """train_and_eval(val_year: int, seed: int) -> float  (val env-PCC).

    Inside the callback, build the datasets with val_scheme='lygo',
    lygo_val_year=val_year, seed=seed, train on Year<val_year, and return the
    macro env-PCC on the LYGO val split.
    """
    recs: list[dict] = []
    for y in val_years:
        for s in seeds:
            pcc = float(train_and_eval(val_year=int(y), seed=int(s)))
            recs.append({"val_year": int(y), "seed": int(s), "val_env_pcc": pcc})

    arr = np.asarray([r["val_env_pcc"] for r in recs], dtype=float)
    per_fold = {}
    for y in sorted({r["val_year"] for r in recs}):
        fa = np.asarray([r["val_env_pcc"] for r in recs if r["val_year"] == y], float)
        per_fold[int(y)] = {"mean": float(fa.mean()),
                            "std": float(fa.std(ddof=1)) if fa.size > 1 else 0.0,
                            "n": int(fa.size)}
    summary = {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
        "per_fold": per_fold,
    }
    return summary, recs


# ============================================================================
# (A) ADD TO utils/dataset.py — GxE_Dataset.__init__
# ----------------------------------------------------------------------------
# A.1  Store params in __init__ (next to the proxy_* params):
#
#     self.lygo_val_year      = lygo_val_year            # None -> latest Year<2024
#     self.lygo_testers       = set(lygo_testers) if lygo_testers else None
#     self.lygo_novelty       = str(lygo_novelty).strip().lower()   # 'parent1'|'hybrid'
#     self.lygo_min_val_per_env = int(lygo_min_val_per_env)
#     self.lygo_valid_val_envs = set(lygo_valid_val_envs) if lygo_valid_val_envs else None
#     self.lygo_info = None
#
# A.2  Add 'lygo' to the val_scheme guard:
#
#     if self.val_scheme not in {"year", "leo", "proxy_same_tester", "lygo"}:
#         raise ValueError(f"Unsupported val_scheme='{val_scheme}'")
#
# A.3  Insert this branch in the SPLIT MASK section, right after the
#      proxy_same_tester branch and before the final `else:` (year default).
#      `x_raw` already has Year/Hybrid/parent1/parent2 derived at this point.
#
#     elif self.val_scheme == "lygo" and split in ("train", "val"):
#         from utils.lygo import (
#             compute_test_testers, compute_lygo_holdout, lygo_split_masks,
#         )
#         # resolve val_year (default: latest pre-test year)
#         if self.lygo_val_year is None:
#             self.lygo_val_year = int(x_raw.loc[x_raw["Year"] < 2024, "Year"].max())
#         # resolve testers (default: the actual 2024 test testers)
#         if self.lygo_testers is None:
#             self.lygo_testers = compute_test_testers(data_path, test_year=2024)
#             if not self.lygo_testers:                      # X_test absent -> safe default
#                 self.lygo_testers = {"PHP02", "LH287"}
#         # resolve held-out envs (compute once on train, reuse on val)
#         if self.lygo_valid_val_envs is None:
#             self.lygo_valid_val_envs, self.lygo_info = compute_lygo_holdout(
#                 x_raw, self.lygo_val_year, self.lygo_testers,
#                 novelty=self.lygo_novelty,
#                 min_val_per_env=self.lygo_min_val_per_env,
#                 test_testers=self.lygo_testers,
#             )
#         train_mask, val_mask = lygo_split_masks(
#             x_raw, self.lygo_val_year, self.lygo_testers,
#             self.lygo_novelty, self.lygo_valid_val_envs,
#         )
#         keep_mask = train_mask if split == "train" else val_mask
#
#   NOTE: split == "test" is unaffected — it falls through to the existing
#   `else` and keeps `Year >= 2024` (the full real 2024 test set). Correct: you
#   evaluate the retrained-on-full-data model there, per the two-stage usage.
#
# A.4  Add the new kwargs to GxE_Dataset.__init__ signature (all default None/...):
#         lygo_val_year=None, lygo_testers=None, lygo_novelty="parent1",
#         lygo_min_val_per_env=4, lygo_valid_val_envs=None,
#
# ============================================================================
# (B) ADD TO utils/utils.py — parse_args  (mirrors the proxy_* args)
# ----------------------------------------------------------------------------
#     p.add_argument("--lygo_val_year", type=int, default=None,
#                    help="LYGO held-out year (default: latest year < 2024).")
#     p.add_argument("--lygo_testers", type=str, default=None,
#                    help="Comma-separated testers; default: derive from 2024 X_test.")
#     p.add_argument("--lygo_novelty", type=str, default="parent1",
#                    choices=["parent1", "hybrid"],
#                    help="Novelty axis for the val set.")
#     p.add_argument("--lygo_min_val_per_env", type=int, default=4,
#                    help="Drop val envs with fewer novel-cross rows (PCC stability).")
#   And extend the val_scheme choices:
#     p.add_argument("--val_scheme", ..., choices=["year","leo","proxy_same_tester","lygo"])
#   Parse the tester string in main(): 
#     lygo_testers = (set(s.strip() for s in args.lygo_testers.split(","))
#                     if args.lygo_testers else None)
#
# ============================================================================
# (C) ADD TO utils/utils.py — make_run_name  (next to `valtag`)
# ----------------------------------------------------------------------------
#     elif val_scheme == "lygo":
#         y = getattr(args, "lygo_val_year", None)
#         valtag = f"lygo{y}+" if y else "lygo+"
#
# ============================================================================
# (D) ADD TO scripts/train.py — dataset construction  (mirrors leo/proxy wiring)
# ----------------------------------------------------------------------------
#   In the train_ds = GxE_Dataset(... ) call, pass:
#         lygo_val_year=lygo_val_year,
#         lygo_testers=lygo_testers,
#         lygo_novelty=lygo_novelty,
#         lygo_min_val_per_env=lygo_min_val_per_env,
#   After train_ds is built, capture the resolved holdout:
#         lygo_val_year   = train_ds.lygo_val_year
#         lygo_testers    = train_ds.lygo_testers
#         lygo_valid_envs = train_ds.lygo_valid_val_envs
#         if is_main(rank) and val_scheme == "lygo":
#             print(f"[INFO] LYGO: {json.dumps(train_ds.lygo_info, sort_keys=True)}")
#   In the val_ds = GxE_Dataset(... ) call, pass the resolved values so val agrees:
#         lygo_val_year=lygo_val_year,
#         lygo_testers=lygo_testers,
#         lygo_novelty=lygo_novelty,
#         lygo_min_val_per_env=lygo_min_val_per_env,
#         lygo_valid_val_envs=lygo_valid_envs,
#   IMPORTANT: keep using the env-PCC eval on the val split with min_samples >= 4
#   for the selector number (matches the env filter above).
# ============================================================================


# ----------------------------------------------------------------------------
# Unit test (pure-function; no file I/O). Mirrors tests/test_gxe_plan.py style.
# Run: python -m pytest utils/lygo.py  (or copy the test into tests/).
# ----------------------------------------------------------------------------
def _synthetic_x_raw() -> pd.DataFrame:
    """Years 2019-2023. parent1 'OLD*' appear from 2019; 'NEW*' debut only in 2023.
    Tester PHP02/LH287 are 'known'; FOO is a non-target tester."""
    rows = []
    rid = 0

    def add(year, loc, p1, p2):
        nonlocal rid
        rid += 1
        env = f"{loc}_{year}"
        rows.append({"id": f"{env}-{p1}/{p2}", "Env": env})

    # 2019-2022: OLD parent1s with known testers, several envs
    for year in (2019, 2020, 2021, 2022):
        for loc in ("LOCA", "LOCB"):
            for i in range(6):
                add(year, loc, f"OLD{i}", "PHP02")
                add(year, loc, f"OLD{i}", "LH287")
    # 2023: NEW parent1s (novel) x known testers -> should be VAL; >=4 per env
    for loc in ("LOCA", "LOCB"):
        for i in range(5):
            add(2023, loc, f"NEW{i}", "PHP02")
    # 2023: an OLD parent1 (known) -> excluded; and a NEW1 x non-target tester -> excluded
    add(2023, "LOCA", "OLD0", "PHP02")
    add(2023, "LOCA", "NEW1", "FOO")
    return pd.DataFrame(rows)


def test_lygo_split():
    x = _synthetic_x_raw()
    valid_envs, info = compute_lygo_holdout(
        x, val_year=2023, testers={"PHP02", "LH287"},
        novelty="parent1", min_val_per_env=4,
    )
    train_mask, val_mask = lygo_split_masks(
        x, 2023, {"PHP02", "LH287"}, "parent1", valid_envs
    )
    xm = _ensure_meta_cols(x)
    tr, va = xm[train_mask.values], xm[val_mask.values]

    assert len(va) > 0, "empty val set"
    assert len(tr) > 0, "empty train set"
    # no row overlap
    assert set(tr["id"]) & set(va["id"]) == set()
    # genetic novelty: no val parent1 seen in training
    assert set(va["parent1"]) & set(tr["parent1"]) == set()
    # temporal novelty: train strictly before val_year, val exactly the val_year
    assert (tr["Year"] < 2023).all()
    assert (va["Year"] == 2023).all()
    # known tester only; non-target tester excluded
    assert set(va["parent2"]).issubset({"PHP02", "LH287"})
    assert "FOO" not in set(va["parent2"])
    # known-parent1 row in 2023 (OLD0) is excluded from val
    assert "OLD0" not in set(va["parent1"])
    assert info["overlap_val_train_parent1s"] == 0
    print("LYGO split OK:", {k: info[k] for k in
          ("n_val_rows", "n_val_envs", "n_val_parent1s", "train_years")})


if __name__ == "__main__":
    test_lygo_split()

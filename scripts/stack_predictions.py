#!/usr/bin/env python
"""
Nonnegative prediction stacker for PCC-first GP-Transformer evaluation.

Fit weights on out-of-fold prediction CSVs, typically rolling-year predictions
from 2020-2023, then apply the weights once to held-out 2024 prediction CSVs.
Input files must contain Env, id, Actual, and Pred. The eval.py scored exports
also include model_name and checkpoint_dir, which are used for provenance.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Prediction files not found: {missing}")
    return paths


def _env_year(env: str) -> int | None:
    m = re.search(r"(20\d{2})", str(env))
    return int(m.group(1)) if m else None


def _safe_pcc(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[valid]
    pred = pred[valid]
    if actual.size < 2 or np.allclose(actual, actual[0]) or np.allclose(pred, pred[0]):
        return np.nan
    return float(np.corrcoef(actual, pred)[0, 1])


def _macro_env_pcc(actual: np.ndarray, pred: np.ndarray, env: np.ndarray) -> float:
    values = []
    for e in pd.Series(env).astype(str).unique():
        mask = env.astype(str) == e
        r = _safe_pcc(actual[mask], pred[mask])
        if np.isfinite(r):
            values.append(r)
    return float(np.mean(values)) if values else np.nan


def _macro_env_mse(actual: np.ndarray, pred: np.ndarray, env: np.ndarray) -> float:
    values = []
    for e in pd.Series(env).astype(str).unique():
        mask = env.astype(str) == e
        if mask.sum():
            values.append(float(np.mean((actual[mask] - pred[mask]) ** 2)))
    return float(np.mean(values)) if values else np.nan


def _model_name(df: pd.DataFrame, path: Path, used: set[str]) -> str:
    if "model_name" in df.columns:
        vals = [v for v in df["model_name"].dropna().astype(str).unique() if v]
        if len(vals) == 1:
            name = vals[0]
        else:
            name = path.stem
    else:
        name = path.stem
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_") or path.stem
    base = name
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _read_prediction_file(path: Path, used: set[str]) -> tuple[str, pd.DataFrame, str]:
    df = pd.read_csv(path)
    required = {"Env", "id", "Actual", "Pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    name = _model_name(df, path, used)
    checkpoint_dir = ""
    if "checkpoint_dir" in df.columns:
        vals = [v for v in df["checkpoint_dir"].dropna().astype(str).unique() if v]
        checkpoint_dir = vals[0] if len(vals) == 1 else ""
    keep = df[["Env", "id", "Actual", "Pred"]].copy()
    keep["Env"] = keep["Env"].astype(str)
    keep["id"] = keep["id"].astype(str)
    keep = keep.rename(columns={"Pred": name, "Actual": f"Actual__{name}"})
    return name, keep, checkpoint_dir


def _merge_predictions(paths: list[Path]) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    used: set[str] = set()
    names: list[str] = []
    provenance: dict[str, str] = {}
    merged: pd.DataFrame | None = None

    for path in paths:
        name, df, checkpoint_dir = _read_prediction_file(path, used)
        names.append(name)
        provenance[name] = checkpoint_dir
        merged = df if merged is None else merged.merge(df, on=["Env", "id"], how="inner")

    if merged is None or merged.empty:
        raise ValueError("No overlapping predictions after merge.")

    actual_cols = [c for c in merged.columns if c.startswith("Actual__")]
    actual = merged[actual_cols[0]].to_numpy(dtype=float)
    for col in actual_cols[1:]:
        other = merged[col].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(other)
        if valid.any() and not np.allclose(actual[valid], other[valid], atol=1e-6):
            raise ValueError(f"Actual labels differ between prediction files at column {col}.")
    merged["Actual"] = actual
    return merged[["Env", "id", "Actual", *names]], names, provenance


def _filter_years(df: pd.DataFrame, year_min: int | None, year_max: int | None) -> pd.DataFrame:
    if year_min is None and year_max is None:
        return df
    years = pd.to_numeric(df["Env"].map(_env_year), errors="coerce")
    mask = pd.Series(True, index=df.index)
    if year_min is not None:
        mask &= years >= year_min
    if year_max is not None:
        mask &= years <= year_max
    return df.loc[mask.fillna(False)].reset_index(drop=True)


def _fit_weights(df: pd.DataFrame, names: list[str], seed: int, n_random: int) -> tuple[np.ndarray, float]:
    y = df["Actual"].to_numpy(dtype=float)
    env = df["Env"].to_numpy(dtype=str)
    preds = df[names].to_numpy(dtype=float)
    n = len(names)

    def score(w: np.ndarray) -> float:
        pred = preds @ w
        r = _macro_env_pcc(y, pred, env)
        return r if np.isfinite(r) else -np.inf

    rng = np.random.default_rng(seed)
    candidates = [np.full(n, 1.0 / n)]
    candidates.extend(np.eye(n))
    if n_random > 0:
        candidates.extend(rng.dirichlet(np.ones(n), size=n_random))

    best_w = max(candidates, key=score)
    best_score = score(best_w)

    try:
        from scipy.optimize import minimize

        def objective(w):
            return -score(np.asarray(w, dtype=float))

        result = minimize(
            objective,
            best_w,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if result.success:
            opt_w = np.asarray(result.x, dtype=float)
            opt_w = np.clip(opt_w, 0.0, None)
            opt_w = opt_w / opt_w.sum()
            opt_score = score(opt_w)
            if opt_score >= best_score:
                best_w, best_score = opt_w, opt_score
    except Exception as exc:
        print(f"[WARN] scipy SLSQP unavailable or failed, using random-search weights: {exc}", file=sys.stderr)

    return best_w, float(best_score)


def _stack_frame(df: pd.DataFrame, names: list[str], weights: np.ndarray, model_name: str) -> pd.DataFrame:
    out = df[["Env", "id", "Actual"]].copy()
    out["Pred"] = df[names].to_numpy(dtype=float) @ weights
    out["model_name"] = model_name
    out["checkpoint_dir"] = "stack:" + ",".join(names)
    return out


def _metrics(df: pd.DataFrame) -> dict[str, float | int]:
    y = df["Actual"].to_numpy(dtype=float)
    p = df["Pred"].to_numpy(dtype=float)
    env = df["Env"].to_numpy(dtype=str)
    return {
        "n_rows": int(len(df)),
        "global_pcc": _safe_pcc(y, p),
        "global_mse": float(np.mean((y - p) ** 2)) if len(df) else np.nan,
        "env_avg_pearson": _macro_env_pcc(y, p, env),
        "env_mse": _macro_env_mse(y, p, env),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-predictions", nargs="+", required=True,
                        help="Out-of-fold prediction CSVs/globs used to fit weights.")
    parser.add_argument("--test-predictions", nargs="+", default=None,
                        help="Held-out prediction CSVs/globs to stack once using fitted weights.")
    parser.add_argument("--train-year-min", type=int, default=2020)
    parser.add_argument("--train-year-max", type=int, default=2023)
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--out-dir", type=Path, default=Path("data/results/stacking"))
    parser.add_argument("--prefix", type=str, default="pcc_stack")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-random", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_paths = _expand_paths(args.train_predictions)
    train_df, names, provenance = _merge_predictions(train_paths)
    train_df = _filter_years(train_df, args.train_year_min, args.train_year_max)
    if train_df.empty:
        raise ValueError("Training prediction merge is empty after year filtering.")

    weights, fit_score = _fit_weights(train_df, names, seed=args.seed, n_random=args.n_random)
    train_stack = _stack_frame(train_df, names, weights, args.prefix)
    train_metrics = _metrics(train_stack)
    train_metrics["fit_env_avg_pearson"] = fit_score

    weights_df = pd.DataFrame({
        "model_name": names,
        "weight": weights,
        "checkpoint_dir": [provenance.get(n, "") for n in names],
    }).sort_values("weight", ascending=False)
    weights_path = args.out_dir / f"{args.prefix}_weights.csv"
    train_path = args.out_dir / f"{args.prefix}_train_stacked_predictions.csv"
    metrics_path = args.out_dir / f"{args.prefix}_metrics.json"
    weights_df.to_csv(weights_path, index=False)
    train_stack.to_csv(train_path, index=False)

    metrics = {"train": train_metrics}
    if args.test_predictions:
        test_paths = _expand_paths(args.test_predictions)
        test_df, test_names, _ = _merge_predictions(test_paths)
        if test_names != names:
            raise ValueError(
                "Test prediction model names/order do not match train predictions. "
                f"train={names}, test={test_names}"
            )
        test_df = _filter_years(test_df, args.test_year, args.test_year)
        if test_df.empty:
            raise ValueError("Test prediction merge is empty after year filtering.")
        test_stack = _stack_frame(test_df, names, weights, args.prefix)
        test_path = args.out_dir / f"{args.prefix}_test_stacked_predictions.csv"
        test_stack.to_csv(test_path, index=False)
        metrics["test"] = _metrics(test_stack)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"[INFO] Wrote weights: {weights_path}")
    print(f"[INFO] Wrote train stack: {train_path}")
    print(f"[INFO] Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()

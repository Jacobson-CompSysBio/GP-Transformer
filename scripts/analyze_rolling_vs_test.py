#!/usr/bin/env python3
"""
Analyze Rolling CV vs 2024 Test Correlation
============================================

Pulls metrics from wandb for a sweep group and computes the Spearman rank
correlation between rolling-CV val metric and 2024-test ``test/env_avg_pearson``.

Works with any sweep (dropout-only, diverse configs, etc.).  For single-fold
sweeps it uses ``cv/mean_val_env_avg_pearson`` (which equals the single fold's
best val); for multi-fold sweeps the same key is the mean across folds.

Usage:
    python scripts/analyze_rolling_vs_test.py --group <SWEEP_GROUP>

    # Fallback — parse from SLURM log files:
    python scripts/analyze_rolling_vs_test.py --from-logs logs/sweeps/<GROUP>-*.out

Requirements:
    pip install wandb scipy  (wandb already installed on Frontier env)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr, kendalltau
except ImportError:
    spearmanr = None
    kendalltau = None


def fetch_from_wandb(
    group: str,
    project: str = "gxe-transformer-rolling",
    entity: str | None = None,
) -> list[dict]:
    """Fetch sweep runs from wandb API, return list of metric dicts."""
    import wandb

    api = wandb.Api(timeout=120)
    entity = entity or os.getenv("WANDB_ENTITY", "jail-ai")

    filters = {"group": group, "state": "finished"}
    runs = api.runs(f"{entity}/{project}", filters=filters, order="+created_at")

    results = []
    for run in runs:
        summary = run.summary._json_dict
        config = run.config

        # Identify config — prefer sweep_tag, fall back to sweep_dropout, then name
        tag = config.get("sweep_tag")
        if not tag:
            m = re.search(r"sweep\+(\S+)", run.name or "")
            tag = m.group(1) if m else run.name or run.id

        row = {
            "run_id": run.id,
            "run_name": run.name,
            "tag": tag,
            "dropout": float(config.get("dropout", 0)),
            "emb_size": int(config.get("emb_size", 0)),
            "g_encoder_type": config.get("g_encoder_type", ""),
            # Rolling CV val metric (works for single or multi-fold)
            "cv_val_pcc": summary.get("cv/mean_val_env_avg_pearson"),
            "cv_val_std": summary.get("cv/std_val_env_avg_pearson"),
            "cv_val_loss": summary.get("cv/mean_val_loss"),
            # Primary test metric
            "test_pcc": summary.get("test/env_avg_pearson"),
            "test_mse": summary.get("test/mse"),
            "test_pcc_weighted": summary.get("test/env_avg_pearson_weighted"),
        }

        # Per-fold test results (if present)
        for year in range(2015, 2024):
            key = f"rolling/test/fold_{year}/env_avg_pearson"
            val = summary.get(key)
            if val is not None:
                row[f"test_fold_{year}_pcc"] = val

        results.append(row)

    return results


def parse_from_logs(log_files: list[str]) -> list[dict]:
    """Fallback: parse metrics from SLURM stdout logs."""
    results = []

    for fpath in log_files:
        text = Path(fpath).read_text(errors="replace")

        # Extract tag from filename (e.g., ...-baseline-12345.out)
        tag_match = re.search(r"-(baseline|tiny|huge_drop|no_moe|zero_drop|drop[\d.]+)-", fpath)
        tag = tag_match.group(1) if tag_match else Path(fpath).stem

        row = {"log_file": fpath, "tag": tag}

        # Extract dropout
        m = re.search(r"DROPOUT=([\d.]+)", text) or re.search(r"dropout.*?=\s*([\d.]+)", text)
        if m:
            row["dropout"] = float(m.group(1))

        # Parse CV summary — single fold logs the same key as mean
        for pattern, key in [
            (r"cv/mean_val_env_avg_pearson[^\d]*([\d.]+)", "cv_val_pcc"),
            (r"cv/mean_val_loss[^\d]*([\d.]+)", "cv_val_loss"),
        ]:
            m = re.search(pattern, text)
            if m:
                row[key] = float(m.group(1))

        # Parse test eval lines: [TEST] <tag>: env_avg_pearson=X.XXXXX, ...mse=Y.YYYYY
        for match in re.finditer(
            r"\[TEST\]\s+(\S+):\s+env_avg_pearson=([\d.]+).*?mse=([\d.]+)",
            text,
        ):
            eval_tag = match.group(1)
            env_pcc = float(match.group(2))
            mse = float(match.group(3))
            if eval_tag in ("best_fold",) or "test_pcc" not in row:
                row["test_pcc"] = env_pcc
                row["test_mse"] = mse
            if "fold_" in eval_tag:
                year = eval_tag.replace("fold_", "")
                row[f"test_fold_{year}_pcc"] = env_pcc

        if "test_pcc" not in row:
            print(f"[WARN] No test metrics found in {fpath}")
            continue

        results.append(row)

    return results


def _safe_rank_corr(x, y, method="spearman"):
    """Compute rank correlation, handling missing scipy gracefully."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), n

    if spearmanr is not None and method == "spearman":
        rho, p = spearmanr(x, y)
        return float(rho), float(p), n

    if kendalltau is not None and method == "kendall":
        tau, p = kendalltau(x, y)
        return float(tau), float(p), n

    # Manual Spearman if scipy not available
    from collections import defaultdict

    def _rank(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    rx, ry = _rank(x), _rank(y)
    d = rx - ry
    rho = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
    return float(rho), float("nan"), n


def print_results(results: list[dict]) -> None:
    """Print formatted results table and correlations."""
    if not results:
        print("\n[ERROR] No results to analyze. Check that the sweep jobs have finished.")
        sys.exit(1)

    # Sort by tag name for stable display
    results = sorted(results, key=lambda r: r.get("tag", ""))
    n = len(results)

    def _fmt(v, w=10):
        return f"{v:{w}.5f}" if v is not None and math.isfinite(v) else f"{'N/A':>{w}s}"

    # ── Table ──
    print()
    print("=" * 90)
    print("Rolling CV vs 2024 Test — Sweep Results")
    print("=" * 90)
    print(
        f"{'Config':<14s}  "
        f"{'CV Val PCC':>10s}  "
        f"{'Test PCC':>10s}  "
        f"{'Test MSE':>10s}  "
        f"{'Drop':>6s}  "
        f"{'Emb':>5s}  "
        f"{'Encoder':>8s}"
    )
    print("-" * 90)

    cv_vals = []
    test_vals = []
    tags = []

    for r in results:
        cv_pcc = r.get("cv_val_pcc")
        test_pcc = r.get("test_pcc")

        cv_vals.append(cv_pcc if cv_pcc is not None else float("nan"))
        test_vals.append(test_pcc if test_pcc is not None else float("nan"))
        tags.append(r.get("tag", "?"))

        print(
            f"{r.get('tag', '?'):<14s}  "
            f"{_fmt(cv_pcc)}  "
            f"{_fmt(test_pcc)}  "
            f"{_fmt(r.get('test_mse'))}  "
            f"{r.get('dropout', '?'):>6}  "
            f"{r.get('emb_size', '?'):>5}  "
            f"{r.get('g_encoder_type', '?'):>8s}"
        )

    print("-" * 90)

    # ── Rank correlations ──
    cv_arr = np.array(cv_vals)
    test_arr = np.array(test_vals)

    print()
    print("Rank Correlations (Rolling CV val PCC vs Test PCC):")
    print("-" * 60)

    rho_s, p_s, n_s = _safe_rank_corr(cv_arr, test_arr, "spearman")
    print(f"  Spearman ρ = {rho_s:+.4f}  (p = {p_s:.4f}, n = {n_s})")

    rho_k, p_k, n_k = _safe_rank_corr(cv_arr, test_arr, "kendall")
    print(f"  Kendall  τ = {rho_k:+.4f}  (p = {p_k:.4f}, n = {n_k})")

    mask = np.isfinite(cv_arr) & np.isfinite(test_arr)
    if mask.sum() >= 3:
        r_pearson = np.corrcoef(cv_arr[mask], test_arr[mask])[0, 1]
        print(f"  Pearson  r = {r_pearson:+.4f}  (n = {mask.sum()})")

    # ── Verdict ──
    print()
    print("=" * 60)
    if math.isnan(rho_s):
        print("VERDICT: Insufficient data to compute correlation.")
    elif rho_s >= 0.8:
        print(f"VERDICT: STRONG (ρ={rho_s:.3f})  →  Rolling CV is a reliable proxy.")
    elif rho_s >= 0.5:
        print(f"VERDICT: MODERATE (ρ={rho_s:.3f})  →  Directionally useful.")
    else:
        print(f"VERDICT: WEAK/NONE (ρ={rho_s:.3f})  →  Rolling CV does NOT track test.")
    print("=" * 60)

    # ── Ranking concordance ──
    if n >= 3:
        cv_rank = np.argsort(np.argsort(-cv_arr))  # 0 = best
        test_rank = np.argsort(np.argsort(-test_arr))  # 0 = best

        print()
        print("Ranking comparison (0 = best):")
        print(f"{'Config':<14s}  {'CV Rank':>8s}  {'Test Rank':>10s}")
        print("-" * 36)
        for i in range(n):
            print(f"{tags[i]:<14s}  {cv_rank[i]:8d}  {test_rank[i]:10d}")

        cv_best_idx = int(np.argmin(cv_rank))
        cv_worst_idx = int(np.argmax(cv_rank))
        print()
        print(f"CV-best  ({tags[cv_best_idx]}) -> test rank {test_rank[cv_best_idx]}")
        print(f"CV-worst ({tags[cv_worst_idx]}) -> test rank {test_rank[cv_worst_idx]}")

        if test_rank[cv_best_idx] <= 1:
            print("  ✓ CV-best is in test top-2")
        else:
            print("  ✗ CV-best is NOT in test top-2")

        if test_rank[cv_worst_idx] >= n - 2:
            print("  ✓ CV-worst is in test bottom-2")
        else:
            print("  ✗ CV-worst is NOT in test bottom-2")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rolling CV vs 2024 test correlation from dropout sweep."
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Wandb group name (SWEEP_GROUP) to filter runs.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="gxe-transformer-rolling",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Wandb entity (default: from WANDB_ENTITY env or 'jail-ai').",
    )
    parser.add_argument(
        "--from-logs",
        nargs="+",
        default=None,
        help="Parse from SLURM log files instead of wandb API.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also dump raw results as JSON.",
    )

    args = parser.parse_args()

    if args.from_logs:
        results = parse_from_logs(args.from_logs)
    elif args.group:
        results = fetch_from_wandb(
            group=args.group,
            project=args.project,
            entity=args.entity,
        )
    else:
        parser.error("Provide either --group (wandb) or --from-logs (log files).")

    if args.json:
        print(json.dumps(results, indent=2, default=str))

    print_results(results)


if __name__ == "__main__":
    main()

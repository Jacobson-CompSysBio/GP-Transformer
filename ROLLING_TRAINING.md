# Rolling Training Quick Guide

This document explains the `ROLLING_*` variables used by:
- `train_rolling.slurm`
- `debug_rolling.slurm`

## Why identical model hyperparameters can still score differently

Even with the same architecture and optimizer settings, rolling CV can produce lower metrics than a single split run because:
- Training data is smaller in early folds (less history available).
- Validation year changes by fold; some years are harder than others.
- Early stopping/checkpoint point differs per fold due to different learning dynamics.
- Aggregate CV metric is an average over multiple temporal conditions, not one fixed split.

So "same hyperparameters" does not imply "same effective training problem."

## Fold construction variables

- `ROLLING_TRAIN_START_YEAR`:
  - First train-max year in forward chain.
- `ROLLING_TRAIN_END_YEAR`:
  - Last validation year endpoint (exclusive in chain construction).
  - Folds are built for `t = start ... end-1` as `(train<=t, val=t+1)`.
- `ROLLING_FULL_CV`:
  - `True`: use all forward folds from `start` to `end`.
  - `False`: use `ROLLING_VAL_YEARS` (or default forward range if empty).
- `ROLLING_VAL_YEARS`:
  - Comma-separated validation years, e.g. `"2021,2022,2023"`.
- `ROLLING_RECENT_FIRST`:
  - If `True`, reverse fold order before truncation.
- `ROLLING_MAX_FOLDS`:
  - Limit to first `K` folds after ordering.
- `ROLLING_SINGLE_VAL_YEAR`:
  - Run exactly one fold for this validation year.
  - Overrides multi-fold behavior.

## Parallel outer-fold variables

- `ROLLING_PARALLEL_OUTER_FOLDS`:
  - `True` means one fold per SLURM array task.
  - Requires array submission (`sbatch --array=...`).
- `ROLLING_PARALLEL_VAL_YEARS`:
  - Optional comma list of val years used for array index mapping.
  - If empty, script falls back to `ROLLING_VAL_YEARS`, then full forward range.
- `ROLLING_PARALLEL_MAX_CONCURRENT`:
  - Used in script guidance text as `%N` concurrency hint.

## Rolling test-eval variables

- `ROLLING_TEST_EVAL_MODE`:
  - `none | best_fold | latest_fold | best_and_latest | all_folds | year:YYYY`
- `ROLLING_TEST_BATCH_SIZE`:
  - Batch size for test evaluation pass.
- `ROLLING_TEST_PRIMARY`:
  - Which evaluated checkpoint becomes canonical `test/*` metrics.

Recommended practice:
- During model selection: `ROLLING_TEST_EVAL_MODE=none`
- Final selected configuration run: enable test eval (`best_fold` is a practical default)

## Common launch patterns

### 1) Single-job rolling CV (no outer parallelism)

Good for strict submission limits:

```bash
sbatch -q extended -t 24:00:00 train_rolling.slurm
```

### 2) Outer-fold parallel array run

Runs folds concurrently (subject to scheduler limits):

```bash
sbatch --export=ALL,ROLLING_PARALLEL_OUTER_FOLDS=True --array=0-2%2 debug_rolling.slurm
```

### 3) Full forward CV range in one job

Set:
- `ROLLING_FULL_CV=True`
- `ROLLING_MAX_FOLDS=""`
- `ROLLING_PARALLEL_OUTER_FOLDS=False`

Then submit normally.


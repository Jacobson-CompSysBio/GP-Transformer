# AGENTS.md - GP-Transformer

Working guidance for agents editing this repo. Scope: maize grain-yield genomic
prediction on the G2F data, scored by macro-averaged per-environment Pearson
correlation on the 2024 test environments.

Treat the current checkout as the source of truth. `shared_memory/` is valuable
project history, but several notes describe branch-only work or old artifacts that
may not match the current checkout.

## Source Of Truth

Read these before substantial work:

- `README.md` for the high-level model and training surface.
- All files in `shared_memory/`, especially:
  - `CONTRASTIVE_PROGRESS.md`
  - `VALIDATION_FAILURE_ANALYSIS.md`
  - `ROLLING_TRAINING.md`
  - `CHANGELOG.affine-shift.md`
- Live code in `scripts/train.py`, `scripts/eval.py`, `utils/dataset.py`,
  `utils/loss.py`, `utils/utils.py`, and `models/model.py`.
- Current launch scripts, especially `best_train.slurm`, `train.slurm`,
  `train_residual.slurm`, and `train_rolling.slurm`.

Important drift checks:

- `utils/target_weighted_validation.py` does not exist in this checkout.
- `scripts/train_sinn.py`, `train_sinn.slurm`, and `slurm/sinn_pipeline.sh`
  are present, but are still ablation paths rather than the current winner.
- Current `scripts/train.py` and `scripts/eval.py` expose `calibration_mode`,
  `checkpoint_tag`, `best_leo`, `latest`, and conditional `best_scale`.
  Do not assume older branch-only names such as `hybrid_combo`,
  `checkpoint_metric`, `metric_prefix`, or `best_select` exist without checking.
- Old checkpoint/result directories may contain tags such as `affcal`, `hybval`,
  or `proxy`; do not infer current CLI support from artifact names alone.

## Current Baseline

Current repo-tracked best result from shared memory:

- Test `env_avg_pearson`: `0.4231675863265991`
- Test weighted `env_avg_pearson`: `0.4290236349883752`
- Test `env_avg_mse`: `101.06045675341392`
- Winning family: plain `FullTransformer + envpcc + e-contrastive`

Use `best_train.slurm` as the clean baseline launcher:

- `FULL_TRANSFORMER=True`
- `G_ENCODER_TYPE=moe`
- `MOE_NUM_EXPERTS=8`, `MOE_TOP_K=2`, `MOE_SHARED_EXPERT=True`
- `MOE_EXPERT_HIDDEN_DIM=256`, `MOE_SHARED_EXPERT_HIDDEN_DIM=256`
- `GBS=8192`, `LR=1e-4`, `WEIGHT_DECAY=1e-5`, `DROPOUT=0.15`
- `GXE_LAYERS=1`, `HEADS=4`, `EMB_SIZE=256`
- `SCALE_TARGETS=False`, `G_INPUT_TYPE=tokens`, `ENV_CATEGORICAL_MODE=drop`
- `LOSS=envpcc`, `LOSS_WEIGHTS=1.0`
- `ENV_STRATIFIED=True`, `MIN_SAMPLES_PER_ENV=32`
- `CONTRASTIVE_MODE=e`
- `ENV_CONTRASTIVE_WEIGHT=0.1`, `ENV_CONTRASTIVE_TEMPERATURE=0.5`
- `CONTRASTIVE_WARMUP_EPOCHS=50`, `CONTRASTIVE_RAMP_EPOCHS=50`
- `LEO_VAL=True`, `LEO_VAL_FRACTION=0.15`
- `PROXY_VALIDATION_MODE=same_tester_novel_cross` logs a diagnostic proxy
  split beside LEO; it is not a checkpoint selector.

`train.slurm` is not currently the canonical winner; it is configured as an
`mse` experiment with `LR=8e-5`. Check launch scripts before assuming a file is
the baseline.

Do not use the older paper-level `0.495` claim or older `0.4265` references as
the current reproducible baseline. The current target remains beating the
competition-winning neighborhood around `0.45`, but the repo-backed floor is
about `0.42317`.

## Data And Splits

Current data path is `data/maize_data_2014-2023_vs_2024_v2/`.

`utils/dataset.py` defines the live split behavior:

- Default train: years `<= 2022`
- Default validation: year `2023`
- Test: years `>= 2024`
- With `LEO_VAL=True`: train/validation are split by held-out pre-2024
  environments instead of by year.

Current `X_train.csv` has `id`, `Env`, 2,224 marker columns, and 705 environment
columns. The code hard-codes `N_ENV=705` and derives the marker block as all
feature columns before the final 705 environment columns.

Genotype/environment handling:

- `G_INPUT_TYPE=tokens` uses dosage tokens derived from 0/0.5/1 marker values.
- `G_INPUT_TYPE=grm` uses train-fitted standardized dosage features.
- `LD_Encoder` requires `G_INPUT_TYPE=tokens`.
- `ENV_CATEGORICAL_MODE=drop` is the baseline behavior; `onehot` is supported
  but not the winning path.
- Checkpoints must keep `env_scaler`, `y_scalers`, and `marker_stats` when
  applicable; eval rebuilds preprocessing from those payloads.

## Active Training Behavior

`scripts/train.py` is the main non-residual path.

- It supports `FullTransformer` and `GxE_Transformer`.
- It builds losses through `utils/loss.py::build_loss`.
- It exposes contrastive schedule controls through
  `CONTRASTIVE_WARMUP_EPOCHS` and `CONTRASTIVE_RAMP_EPOCHS`, defaulting to
  the historical `50/50` behavior.
- It selects checkpoints by full-validation `val_loss/env_avg_pearson`
  maximization, with `val_loss` as the tie-breaker.
- Validation gathers predictions/targets/env ids across DDP ranks before
  computing metrics, so validation selection is closer to eval-time scoring than
  the local batch training loss.
- Saved checkpoints include numeric improvement checkpoints plus aliases:
  `best_leo.pt`, `latest.pt`, and, for `CALIBRATION_MODE=env_affine`,
  `best_scale.pt`. `scripts/eval.py --checkpoint_tag` resolves these through
  `checkpoint_manifest.json`.

`scripts/train_residual.py` and `train_residual.slurm` are experimental residual
paths. They are not the current winning family.

`scripts/train_rolling.py` and `train_rolling.slurm` implement rolling temporal
CV. Use `shared_memory/ROLLING_TRAINING.md` for variable semantics. Short-budget
rolling CV can rank learning speed instead of converged test quality, so do not
promote a rolling result without matched budget evidence.

## Losses And Metrics

Leaderboard priority is within-environment ranking:

1. Macro environment PCC is primary.
2. Weighted environment PCC and global PCC are diagnostics.
3. MSE/RMSE are scale diagnostics and should not dominate a rank-first trunk.

Live `utils/loss.py` supports:

- `envpcc`, `pcc`, `mse`, `envmse`, `envspearman`, `spearman`, `ktau`, `triplet`
- `xi` is registered only as a placeholder; `build_loss("xi")` raises until
  `XiLoss.forward()` is implemented.

Important implementation details:

- `envwise_pcc` computes local per-batch per-environment PCC and uniformly
  averages valid environments. The docstring and README are aligned with this;
  the current implementation does not Fisher-z average.
- `macro_env_pearson` is the eval-style metric helper used for validation and
  test reporting.
- `ENV_STRATIFIED=True` only swaps in `EnvStratifiedSampler` in `scripts/train.py`
  when `"envpcc"` appears in the loss string.
- `envwise_pcc` falls back to global PCC if no environment in the batch has
  enough valid samples.

Known result guidance:

- Keep `envpcc` as the primary trunk objective unless deliberately running a
  controlled ablation.
- Direct `mse` training under the contrastive setup was much worse for test
  macro env PCC.
- If scale matters, prefer rank-first calibration through
  `CALIBRATION_MODE=env_affine` with `ENVCCC_WEIGHT`/`HUBER_WEIGHT`, and accept
  it only when PCC loss stays within the matched-run tolerance.

## Validation Reality

The 2024 test requires capabilities no current holdout exactly reproduces:

- 90.2% of test hybrids are novel.
- 53% of test hybrids use tester `LH287`, which has zero training rows.
- 2024 uses `LH287` and `PHP02`; historical holdouts use different tester
  structure.
- `LEO` tests held-out environments with mostly known genotype structure.
- 2023 validation has almost no novel hybrids.
- Target-weighted validation reweighted environment similarity but did not fix
  genotype/tester shift in the documented matched runs.
- Proxy selector work selected worse checkpoints in documented runs. The current
  code supports a same-tester novel-cross proxy diagnostic through
  `PROXY_VALIDATION_MODE=same_tester_novel_cross`, but canonical checkpoint
  selection should stay on LEO until proxy-vs-test ranking is proven across a
  matched run suite.

Use `LEO_VAL=True` because it is the current best-supported selector for the
winning recipe, not because it perfectly mirrors 2024. Any new validation scheme
must be judged by whether it ranks multiple configurations in the same order as
2024 test performance.

Promising validation direction from the repo notes: tester-aware or
same-tester novel-cross holdouts, then optional environment weighting inside
that proxy. Do not lead with 2023-only or environment-only weighting.

## Experiment Priorities

Highest-confidence next steps:

- Sweep the current `e`-contrastive winner locally around weight `0.1` and
  temperature `0.5` before changing the trunk.
- Use `sweep_econtrastive_pcc.sh` phases for baseline repeats, the seed-1
  weight/temperature grid, top repeats, schedule sweep, and calibration sweep.
- Repeat the best recipe across seeds before treating tiny deltas as real.
- Eval now exports prediction-scale diagnostics, per-env/tester/novelty CSVs,
  and scored predictions with `model_name`/`checkpoint_dir` provenance.
- If changing `envwise_pcc`, either update the docstring/README or implement the
  promised Fisher-z behavior and verify validation/test parity.
- Investigate tester-aware validation using `scripts/analyze_holdout_strategies.py`
  as the starting point.

Lower priority unless explicitly requested:

- Three-prong architecture work. It matched or exceeded validation in some
  branch experiments but transferred worse to test than FullTransformer.
- Residual or affine calibration as a primary route to higher PCC. Keep these as
  scale/calibration tools unless paired evidence shows no rank regression.
- SINN/staged structured training. The notes are useful, but the current
  implementation files are absent from `main`.
- Attention pooling and uncertainty-weighting snippets from older guidance.
  Reintroduce them only as normal code changes with tests and ablations.

## Working Rules

- Preserve the current DDP pattern when changing training or validation.
- Keep new W&B metrics under `train_loss/`, `train_loss_epoch/`, `val_loss/`, or
  `test/` namespaces.
- Use `debug.slurm` or an explicitly reduced SLURM config for smoke tests before
  full Frontier runs.
- When comparing losses or samplers, report the exact launcher, run name,
  checkpoint directory, and test metrics CSV.
- Do not stack multiple architecture/loss/validation changes before verifying
  each one against the current `best_train.slurm` baseline.

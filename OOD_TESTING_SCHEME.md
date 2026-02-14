# OOD Testing Scheme

This scheme tests three additions:
- reaction-norm interaction head
- covariate-shift weighted auxiliary loss
- GroupDRO auxiliary loss

Use the same fold/split protocol for all runs (same seed list and same rolling years) to keep comparisons fair.

## Metrics To Track

- Primary: `val/env_avg_pearson` (or rolling `cv/mean_val_env_avg_pearson`)
- Secondary: `val_loss`, `val/env_avg_pearson_weighted` (if logged), stability across folds/seeds
- For OOD diagnostics:
  - `train_loss/shift_mse`
  - `train_loss/group_dro`
  - `train_loss/group_dro_q_max`
  - `train_loss/group_dro_q_min`

## Stage 0: Baseline Lock

Run your current best hyperparameters with all new options off:

- `REACTION_NORM=False`
- `SHIFT_WEIGHTING=False`
- `SHIFT_LOSS_WEIGHT=0.0`
- `GROUP_DRO=False`
- `GROUP_DRO_WEIGHT=0.0`

Use this as the comparison anchor.

## Stage 1: Reaction-Norm Only

Run 2 settings:

1. `REACTION_NORM=True`, `REACTION_NORM_RANK=32`, `REACTION_NORM_WEIGHT=1.0`
2. `REACTION_NORM=True`, `REACTION_NORM_RANK=64`, `REACTION_NORM_WEIGHT=0.5`

Keep all OOD robust losses off.

Decision rule:
- If no gain vs baseline, keep reaction-norm off for later stages.

## Stage 2: Shift Weighting Only

Turn on importance weighting but keep GroupDRO off:

- `SHIFT_WEIGHTING=True`
- `SHIFT_WEIGHT_USE_GENOTYPE=False` (start with env-only, cheaper and usually stable)
- `SHIFT_LOSS_WEIGHT` grid: `0.02`, `0.05`, `0.1`
- `SHIFT_WEIGHT_POWER=1.0`
- `SHIFT_WEIGHT_CLIP_MAX=5.0` (or `10.0` if weights are too flat)

Optional follow-up:
- `SHIFT_WEIGHT_USE_GENOTYPE=True`, `SHIFT_WEIGHT_MARKER_DIM=512`

Decision rule:
- Keep the smallest `SHIFT_LOSS_WEIGHT` that improves CV and does not increase fold variance sharply.

## Stage 3: GroupDRO Only

Turn on GroupDRO and keep shift weighting off:

- `GROUP_DRO=True`
- `GROUP_DRO_GROUP_BY=env` (start here)
- `GROUP_DRO_WEIGHT` grid: `0.02`, `0.05`, `0.1`
- `GROUP_DRO_STEP_SIZE` grid: `0.01`, `0.05`

Optional follow-up:
- `GROUP_DRO_GROUP_BY=year`

Decision rule:
- Prefer settings that improve worst-fold behavior, not only mean.

## Stage 4: Combined Best

Combine the best settings from Stages 1-3:

- reaction-norm best candidate
- shift-weight best candidate
- GroupDRO best candidate

Try 2 runs:

1. reaction + shift
2. reaction + shift + GroupDRO

## Compute-Efficient Recipe

1. Fast screen:
   - shorter run budget (`NUM_EPOCHS` lower, smaller `EARLY_STOP`)
   - recent folds only
2. Confirm:
   - full rolling folds
   - original epoch budget
3. Final:
   - pick top 1-3 configs from rolling CV
   - retrain on all pre-2024
   - evaluate once on 2024 test

## Suggested First Batch (Minimal)

Run these 4 jobs first:

1. Baseline locked
2. Reaction-norm only (`rank=32`, `weight=1.0`)
3. Shift-only (`SHIFT_LOSS_WEIGHT=0.05`)
4. GroupDRO-only (`GROUP_DRO_WEIGHT=0.05`, `GROUP_DRO_GROUP_BY=env`)

Then keep only winners for larger sweeps.

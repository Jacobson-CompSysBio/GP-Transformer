#!/bin/bash
set -euo pipefail

# Dual-specialist wave:
# 1. Lock the current FT rank branch
# 2. Rebuild tester-aware validation evidence
# 3. Train residual absolute-yield specialists

submit_leo() {
    local suffix="$1"
    local model_seed="$2"
    local split_seed="$3"
    local envccc_weight="$4"
    local huber_weight="$5"
    local cal_start="$6"
    local cal_ramp="$7"
    local cal_detach="$8"
    local cal_joint="$9"

    sbatch --export=ALL,\
RUN_NAME_SUFFIX="$suffix",\
MODEL_SEED="$model_seed",\
SPLIT_SEED="$split_seed",\
ENVCCC_WEIGHT="$envccc_weight",\
HUBER_WEIGHT="$huber_weight",\
CALIBRATION_START_EPOCH="$cal_start",\
CALIBRATION_RAMP_EPOCHS="$cal_ramp",\
CALIBRATION_DETACH_RANK_UNTIL_EPOCH="$cal_detach",\
CALIBRATION_JOINT_GRAD_FRACTION="$cal_joint" \
train.slurm
}

submit_hybrid() {
    local suffix="$1"
    local val_scheme="$2"
    local model_seed="$3"
    local split_seed="$4"
    local proxy_seed="$5"
    local proxy_holdout_frac="$6"
    local proxy_year_min="$7"
    local proxy_year_max="$8"
    local envccc_weight="$9"
    local huber_weight="${10}"

    sbatch --export=ALL,\
RUN_NAME_SUFFIX="$suffix",\
VAL_SCHEME="$val_scheme",\
MODEL_SEED="$model_seed",\
SPLIT_SEED="$split_seed",\
PROXY_SEED="$proxy_seed",\
PROXY_HOLDOUT_FRAC="$proxy_holdout_frac",\
PROXY_YEAR_MIN="$proxy_year_min",\
PROXY_YEAR_MAX="$proxy_year_max",\
ENVCCC_WEIGHT="$envccc_weight",\
HUBER_WEIGHT="$huber_weight" \
hybrid_train.slurm
}

submit_residual() {
    local suffix="$1"
    local model_seed="$2"
    local split_seed="$3"
    local val_scheme="$4"
    local leo_val="$5"
    local loss="$6"
    local loss_weights="$7"
    local lambda_ymean="$8"
    local lambda_resid="$9"

    sbatch --export=ALL,\
RUN_NAME_SUFFIX="$suffix",\
MODEL_SEED="$model_seed",\
SPLIT_SEED="$split_seed",\
VAL_SCHEME="$val_scheme",\
LEO_VAL="$leo_val",\
LOSS="$loss",\
LOSS_WEIGHTS="$loss_weights",\
LAMBDA_YMEAN="$lambda_ymean",\
LAMBDA_RESID="$lambda_resid" \
train_residual.slurm
}

echo "Submitting Phase 1: rank specialist lock"
submit_leo ft_leo_envccc005_h02_s2            2 1 0.05 0.02 150 100 250 0.10
submit_leo ft_leo_envccc010_h01_s2            2 1 0.10 0.01 150 100 250 0.10
submit_leo ft_leo_envccc005_h02_frozencal_s1  1 1 0.05 0.02 175 75  9999 0.0
submit_leo ft_leo_envccc005_h02_latecal_s1    1 1 0.05 0.02 200 150 350 0.05

echo "Submitting Phase 2: tester-aware validation evidence"
submit_hybrid ft_proxy_same_tester_recent025_s1 proxy_same_tester 1 1 1 0.25 2022 2023 0.05 0.02
submit_hybrid ft_proxy_same_tester_recent025_s2 proxy_same_tester 2 1 1 0.25 2022 2023 0.05 0.02
submit_hybrid ft_hybrid_recent025_s1            hybrid_combo      1 1 1 0.25 2022 2023 0.05 0.02
submit_hybrid ft_hybrid_recent025_s2            hybrid_combo      2 1 1 0.25 2022 2023 0.05 0.02

echo "Submitting Phase 3: residual RMSE specialists"
submit_residual res_year_mixA_s1   1 1 year False "envpcc+envmse+pcc+mse" "1.0,0.1,0.1,0.1" 0.05 1.0
submit_residual res_year_mixB_s1   1 1 year False "envpcc+envmse+pcc+mse" "0.7,0.2,0.1,0.2" 0.10 1.0
submit_residual res_year_scale_s1  1 1 year False "mse+envmse+pcc"        "0.6,0.3,0.1"     0.20 0.75
submit_residual res_year_mixA_s2   2 1 year False "envpcc+envmse+pcc+mse" "1.0,0.1,0.1,0.1" 0.05 1.0

cat <<'EOF'

Pre-run checklist still required outside this launcher:
  1. Recover the eval-only outputs for timed-out run 4239401 before comparing old vs new proxy tracking.
  2. After this 12-job wave, fit the learned stack from exported prediction CSVs.

EOF

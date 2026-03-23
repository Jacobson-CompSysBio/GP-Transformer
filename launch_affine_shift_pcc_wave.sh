#!/bin/bash
set -euo pipefail

# PCC-first affine-shift experiment wave.
# Batch 1 and Batch 2 are ready to submit as-is.
# Batch 3 is intentionally left as commented templates because the winning
# Batch 1 / Batch 2 configs are not knowable until those runs finish.

submit_leo() {
    local suffix="$1"
    local model_seed="$2"
    local split_seed="$3"
    local envccc_weight="$4"
    local huber_weight="$5"

    sbatch --export=ALL,\
RUN_NAME_SUFFIX="$suffix",\
MODEL_SEED="$model_seed",\
SPLIT_SEED="$split_seed",\
ENVCCC_WEIGHT="$envccc_weight",\
HUBER_WEIGHT="$huber_weight" \
train.slurm
}

submit_hybrid() {
    local suffix="$1"
    local model_seed="$2"
    local split_seed="$3"
    local proxy_seed="$4"
    local proxy_holdout_frac="$5"
    local envccc_weight="$6"
    local huber_weight="$7"

    sbatch --export=ALL,\
RUN_NAME_SUFFIX="$suffix",\
MODEL_SEED="$model_seed",\
SPLIT_SEED="$split_seed",\
PROXY_SEED="$proxy_seed",\
PROXY_HOLDOUT_FRAC="$proxy_holdout_frac",\
ENVCCC_WEIGHT="$envccc_weight",\
HUBER_WEIGHT="$huber_weight" \
hybrid_train.slurm
}

echo "Submitting Batch 1: canonical LEO calibration sweep"
submit_leo batch1-control      1 1 0.10 0.02
submit_leo batch1-huber001     1 1 0.10 0.01
submit_leo batch1-huber005     1 1 0.10 0.05
submit_leo batch1-envccc005    1 1 0.05 0.02

echo "Submitting Batch 2: hybrid tracking sweep"
submit_hybrid batch2-proxy020  1 1 1 0.20 0.10 0.02
submit_hybrid batch2-proxy025  1 1 1 0.25 0.10 0.02

cat <<'EOF'

Batch 3 templates:
  1. Review Batch 1 and identify the winning LEO config.
  2. Review Batch 2 and identify the better hybrid config.
  3. Re-run each with MODEL_SEED=2 and the same SPLIT_SEED=1.

Example:
  sbatch --export=ALL,RUN_NAME_SUFFIX=batch3-leo-repeat,MODEL_SEED=2,SPLIT_SEED=1,ENVCCC_WEIGHT=0.10,HUBER_WEIGHT=0.02 train.slurm
  sbatch --export=ALL,RUN_NAME_SUFFIX=batch3-hybrid-repeat,MODEL_SEED=2,SPLIT_SEED=1,PROXY_SEED=1,PROXY_HOLDOUT_FRAC=0.20,ENVCCC_WEIGHT=0.10,HUBER_WEIGHT=0.02 hybrid_train.slurm
EOF

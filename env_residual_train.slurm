#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J env-residual-v1
#SBATCH -N 4
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# This is the retained treatment from job 5033556_1.
export SCALE_TARGETS=True
export DECOMPOSITION_SCALE_MODE=shared_total
export PREDICTION_HEAD=env_residual
export ENV_MEAN_LOSS_WEIGHT=1.0
export ENV_TOTAL_HUBER_WEIGHT=0.05
export CALIBRATION_MODE=none
export PROXY_VALIDATION_MODE=none

bash "$REPO_ROOT/best_train.slurm"

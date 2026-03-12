#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J sinn-e
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

### ═══════════════════════════════════════════════════════════════
### Phase E: Train environment encoder on E_hat
### ═══════════════════════════════════════════════════════════════
set -eo pipefail
ulimit -S -c 0

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

### Shared model config ###
export DECOMP_PATH="data/decomposition/decomposition_2022.json"
export DECOMP_SOURCE=split
export G_INPUT_TYPE=tokens
export ENV_CATEGORICAL_MODE=drop
export SEED=1
export SCALE_TARGETS=False
export LEO_VAL=False
export LEO_VAL_FRACTION=0.10
export E_SELECT_METRIC=mse_minus_corr
export E_CORR_WEIGHT=5.0

export EMB_SIZE=256
export HEADS=4
export G_LAYERS=1
export MLP_LAYERS=1
export E_MLP_LAYERS=1
export E_HIDDEN_DIM=8
export GXE_LAYERS=2
export DROPOUT=0.1
export E_DROPOUT=0.1

### Phase E hyperparameters ###
E_LR=3e-4
E_EPOCHS=500
E_EARLY_STOP=15
E_BATCH_SIZE=256

### Compute ###
NGPUS=$(($SLURM_NNODES * 8))
export GBS=8192

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0

export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

mkdir -p logs/ckpt_ids logs/run_ids checkpoints data/results

SINN_TAG="${SINN_TAG:-${SLURM_JOB_ID}}"
SINN_CKPT_DIR="checkpoints/sinn-${SINN_TAG}"
mkdir -p "$SINN_CKPT_DIR"

echo "[SINN] Phase E — Job ${SLURM_JOB_ID}, Tag ${SINN_TAG}"

export WANDB_RUN_ID_FILE="logs/run_ids/wandb_sinn_e_${SLURM_JOB_ID}.txt"
export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_sinn_e_${SLURM_JOB_ID}.txt"
rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"

srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train_sinn.py \
        --sinn_phase e \
        --decomp_path "$DECOMP_PATH" \
        --decomp_source $DECOMP_SOURCE \
        --batch_size $E_BATCH_SIZE \
        --gbs $GBS \
        --lr $E_LR \
        --num_epochs $E_EPOCHS \
        --early_stop $E_EARLY_STOP \
        --g_input_type $G_INPUT_TYPE \
        --env_categorical_mode $ENV_CATEGORICAL_MODE \
        --emb_size $EMB_SIZE \
        --heads $HEADS \
        --g_layers $G_LAYERS \
        --mlp_layers $MLP_LAYERS \
        --e_mlp_layers $E_MLP_LAYERS \
        --e_hidden_dim $E_HIDDEN_DIM \
        --gxe_layers $GXE_LAYERS \
        --dropout $DROPOUT \
        --e_dropout $E_DROPOUT \
        --seed $SEED \
        --scale_targets $SCALE_TARGETS \
        --leo_val $LEO_VAL \
        --leo_val_fraction $LEO_VAL_FRACTION \
        --e_select_metric $E_SELECT_METRIC \
        --e_corr_weight $E_CORR_WEIGHT

if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
    E_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
    E_CKPT_PATH="${E_CKPT_DIR}/best_e.pt"
    echo "[SINN] Phase E checkpoint: $E_CKPT_PATH"
    cp "$E_CKPT_PATH" "${SINN_CKPT_DIR}/best_e.pt" 2>/dev/null || true

    echo "$E_CKPT_PATH" > "logs/ckpt_ids/sinn_e_ckpt_${SINN_TAG}.txt"
    echo "[SINN] Phase E complete. Downstream scripts can find ckpt via SINN_TAG=${SINN_TAG}"
else
    echo "[ERROR] Phase E checkpoint dir not found"
    exit 1
fi

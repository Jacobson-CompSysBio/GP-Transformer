#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J sinn-ge
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

### ═══════════════════════════════════════════════════════════════
### Phase GE: Train interaction model on GE_hat
### Requires: G and E checkpoints from prior phases
###   Set SINN_TAG to find them automatically, or pass:
###     G_CKPT_PATH=/path/to/best_g.pt
###     E_CKPT_PATH=/path/to/best_e.pt
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
export G_INPUT_TYPE=tokens
export ENV_CATEGORICAL_MODE=drop
export SEED=1
export SCALE_TARGETS=False
export LEO_VAL=True
export LEO_VAL_FRACTION=0.10

export EMB_SIZE=512
export HEADS=4
export G_LAYERS=1
export MLP_LAYERS=1
export E_MLP_LAYERS=1
export GXE_LAYERS=2
export DROPOUT=0.1
export E_DROPOUT=0.25

# MoE settings
export FULL_TF_MLP_TYPE=moe
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=2
export MOE_SHARED_EXPERT=True
export MOE_EXPERT_HIDDEN_DIM=512
export MOE_SHARED_EXPERT_HIDDEN_DIM=512
export MOE_LOSS_WEIGHT=0.01

### Phase GE hyperparameters ###
GE_LR=5e-4
GE_ENCODER_LR=5e-5
GE_EPOCHS=500
GE_EARLY_STOP=100
GE_BATCH_SIZE=256
FREEZE_ENCODERS=True

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

echo "[SINN] Phase GE — Job ${SLURM_JOB_ID}, Tag ${SINN_TAG}"

### Resolve G and E checkpoints ###
if [[ -z "$G_CKPT_PATH" ]]; then
    G_CKPT_FILE="logs/ckpt_ids/sinn_g_ckpt_${SINN_TAG}.txt"
    if [[ -f "$G_CKPT_FILE" ]]; then
        G_CKPT_PATH=$(cat "$G_CKPT_FILE")
    fi
fi
if [[ -z "$E_CKPT_PATH" ]]; then
    E_CKPT_FILE="logs/ckpt_ids/sinn_e_ckpt_${SINN_TAG}.txt"
    if [[ -f "$E_CKPT_FILE" ]]; then
        E_CKPT_PATH=$(cat "$E_CKPT_FILE")
    fi
fi

GE_EXTRA_ARGS=""
if [[ -n "$G_CKPT_PATH" && -f "$G_CKPT_PATH" ]]; then
    GE_EXTRA_ARGS="$GE_EXTRA_ARGS --g_ckpt $G_CKPT_PATH"
    echo "[SINN] Using G checkpoint: $G_CKPT_PATH"
else
    echo "[WARN] No G checkpoint found — training GE from scratch"
fi
if [[ -n "$E_CKPT_PATH" && -f "$E_CKPT_PATH" ]]; then
    GE_EXTRA_ARGS="$GE_EXTRA_ARGS --e_ckpt $E_CKPT_PATH"
    echo "[SINN] Using E checkpoint: $E_CKPT_PATH"
else
    echo "[WARN] No E checkpoint found — training GE from scratch"
fi

export WANDB_RUN_ID_FILE="logs/run_ids/wandb_sinn_ge_${SLURM_JOB_ID}.txt"
export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_sinn_ge_${SLURM_JOB_ID}.txt"
rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"

srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train_sinn.py \
        --sinn_phase ge \
        --decomp_path "$DECOMP_PATH" \
        --batch_size $GE_BATCH_SIZE \
        --gbs $GBS \
        --lr $GE_LR \
        --num_epochs $GE_EPOCHS \
        --early_stop $GE_EARLY_STOP \
        --g_input_type $G_INPUT_TYPE \
        --env_categorical_mode $ENV_CATEGORICAL_MODE \
        --emb_size $EMB_SIZE \
        --heads $HEADS \
        --g_layers $G_LAYERS \
        --mlp_layers $MLP_LAYERS \
        --e_mlp_layers $E_MLP_LAYERS \
        --gxe_layers $GXE_LAYERS \
        --dropout $DROPOUT \
        --e_dropout $E_DROPOUT \
        --seed $SEED \
        --scale_targets $SCALE_TARGETS \
        --leo_val $LEO_VAL \
        --leo_val_fraction $LEO_VAL_FRACTION \
        --freeze_encoders $FREEZE_ENCODERS \
        --ge_encoder_lr $GE_ENCODER_LR \
        $GE_EXTRA_ARGS

if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
    GE_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
    GE_CKPT_PATH="${GE_CKPT_DIR}/best_ge.pt"
    echo "[SINN] Phase GE checkpoint: $GE_CKPT_PATH"
    cp "$GE_CKPT_PATH" "${SINN_CKPT_DIR}/best_ge.pt" 2>/dev/null || true

    echo "$GE_CKPT_PATH" > "logs/ckpt_ids/sinn_ge_ckpt_${SINN_TAG}.txt"
    echo "[SINN] Phase GE complete. Downstream scripts can find ckpt via SINN_TAG=${SINN_TAG}"
else
    echo "[ERROR] Phase GE checkpoint dir not found"
    exit 1
fi

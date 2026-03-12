#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J sinn-ft
#SBATCH -N 4
#SBATCH -t 04:00:00
#SBATCH -p extended
##SBATCH -q debug
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

### ═══════════════════════════════════════════════════════════════
### Phase Finetune: End-to-end on raw yield with envpcc+mse loss
### Requires: GE checkpoint from Phase GE
###   Set SINN_TAG to find it automatically, or pass:
###     GE_CKPT_PATH=/path/to/best_ge.pt
###
### Uses the launcher node count below (currently 4 nodes / 32 GPUs).
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
export LEO_VAL=True
export LEO_VAL_FRACTION=0.10
export SINN_INTERACTION=bilinear

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

### Phase Finetune hyperparameters ###
FT_LR=1e-4
FT_EPOCHS=1000
FT_EARLY_STOP=200
FT_LOSS="envpcc+mse"
FT_LOSS_WEIGHTS="1.0,0.1"
ENV_STRATIFIED=True
MIN_SAMPLES_PER_ENV=32
AUX_G_WEIGHT=0.1
AUX_E_WEIGHT=0.1
AUX_YEAR_WEIGHT=0.1

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

echo "[SINN] Phase Finetune — Job ${SLURM_JOB_ID}, Tag ${SINN_TAG}"
echo "[SINN] Nodes: ${SLURM_NNODES}, GPUs: ${NGPUS}"

### Resolve GE checkpoint ###
if [[ -z "$GE_CKPT_PATH" ]]; then
    GE_CKPT_FILE="logs/ckpt_ids/sinn_ge_ckpt_${SINN_TAG}.txt"
    if [[ -f "$GE_CKPT_FILE" ]]; then
        GE_CKPT_PATH=$(cat "$GE_CKPT_FILE")
    fi
fi

FT_EXTRA_ARGS=""
if [[ -n "$GE_CKPT_PATH" && -f "$GE_CKPT_PATH" ]]; then
    FT_EXTRA_ARGS="--ge_ckpt $GE_CKPT_PATH"
    echo "[SINN] Using GE checkpoint: $GE_CKPT_PATH"
else
    echo "[ERROR] No GE checkpoint found. Set GE_CKPT_PATH or SINN_TAG."
    exit 1
fi

# Microbatch per GPU (more nodes → smaller per-GPU batch → fits in memory)
FT_MBS=$(($GBS / $NGPUS))
echo "[SINN] Microbatch size per GPU: $FT_MBS"

export WANDB_RUN_ID_FILE="logs/run_ids/wandb_sinn_ft_${SLURM_JOB_ID}.txt"
export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_sinn_ft_${SLURM_JOB_ID}.txt"
rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"

srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train_sinn.py \
        --sinn_phase finetune \
        --decomp_path "$DECOMP_PATH" \
        --decomp_source $DECOMP_SOURCE \
        --batch_size $FT_MBS \
        --gbs $GBS \
        --finetune_lr $FT_LR \
        --num_epochs $FT_EPOCHS \
        --early_stop $FT_EARLY_STOP \
        --loss $FT_LOSS \
        --loss_weights "$FT_LOSS_WEIGHTS" \
        --env_stratified $ENV_STRATIFIED \
        --min_samples_per_env $MIN_SAMPLES_PER_ENV \
        --g_input_type $G_INPUT_TYPE \
        --env_categorical_mode $ENV_CATEGORICAL_MODE \
        --emb_size $EMB_SIZE \
        --heads $HEADS \
        --g_layers $G_LAYERS \
        --mlp_layers $MLP_LAYERS \
        --e_mlp_layers $E_MLP_LAYERS \
        --gxe_layers $GXE_LAYERS \
        --dropout $DROPOUT \
        --sinn_interaction $SINN_INTERACTION \
        --e_dropout $E_DROPOUT \
        --seed $SEED \
        --scale_targets $SCALE_TARGETS \
        --leo_val $LEO_VAL \
        --leo_val_fraction $LEO_VAL_FRACTION \
        --aux_g_weight $AUX_G_WEIGHT \
        --aux_e_weight $AUX_E_WEIGHT \
        --aux_year_weight $AUX_YEAR_WEIGHT \
        $FT_EXTRA_ARGS

if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
    FT_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
    FT_CKPT_PATH="${FT_CKPT_DIR}/best_finetune.pt"
    echo "[SINN] Phase Finetune checkpoint: $FT_CKPT_PATH"
    cp "$FT_CKPT_PATH" "${SINN_CKPT_DIR}/best_finetune.pt" 2>/dev/null || true
else
    echo "[ERROR] Phase Finetune checkpoint dir not found"
    exit 1
fi

### ═══════════════════════════════════════════════════════════════
### Eval on test set
### ═══════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo "[SINN] Running evaluation on test set"
echo "============================================"

if [[ -f "$WANDB_RUN_ID_FILE" ]]; then
    export WANDB_RESUME=allow
    export WANDB_RUN_ID=$(cat "$WANDB_RUN_ID_FILE")
fi

if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
    EVAL_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
    python -u scripts/eval.py --checkpoint_dir "$EVAL_CKPT_DIR"
else
    echo "[ERROR] No finetune checkpoint dir found for eval"
fi

echo ""
echo "[SINN] Phase Finetune + Eval complete."
echo "[SINN] Checkpoints in: $SINN_CKPT_DIR"

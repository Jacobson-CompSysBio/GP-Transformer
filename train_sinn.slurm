#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J sinn
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

### env, modules, and settings ###
set -eo pipefail
ulimit -S -c 0

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

### Which phases to run (set to False to skip) ###
RUN_PHASE_G=True
RUN_PHASE_E=True
RUN_PHASE_GE=True
RUN_PHASE_FINETUNE=True

### Decomposition path ###
export DECOMP_PATH="data/decomposition/decomposition_2022.json"

### Shared hyperparameters ###
export G_INPUT_TYPE=tokens
export ENV_CATEGORICAL_MODE=drop
export SEED=1
export SCALE_TARGETS=False
export LEO_VAL=True
export LEO_VAL_FRACTION=0.10

# Model size (shared across phases)
export EMB_SIZE=256
export HEADS=4
export G_LAYERS=1
export MLP_LAYERS=1
export GXE_LAYERS=1
export DROPOUT=0.1

# MoE settings (for ge/finetune FullTransformer)
export FULL_TF_MLP_TYPE=moe
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=2
export MOE_SHARED_EXPERT=True
export MOE_EXPERT_HIDDEN_DIM=256
export MOE_SHARED_EXPERT_HIDDEN_DIM=256
export MOE_LOSS_WEIGHT=0.01

### Phase-specific hyperparameters ###
# Phase g (genotype encoder)
G_LR=1e-3
G_EPOCHS=100
G_EARLY_STOP=25
G_BATCH_SIZE=256

# Phase e (environment encoder)
E_LR=1e-3
E_EPOCHS=500
E_EARLY_STOP=100
E_BATCH_SIZE=256

# Phase ge (interaction)
GE_LR=5e-4
GE_EPOCHS=500
GE_EARLY_STOP=100
GE_BATCH_SIZE=256
FREEZE_ENCODERS=True

# Phase finetune
FT_LR=1e-5
FT_EPOCHS=1000
FT_EARLY_STOP=200
FT_LOSS="envpcc"
FT_LOSS_WEIGHTS="1.0"
ENV_STRATIFIED=True
MIN_SAMPLES_PER_ENV=32

# Compute microbatch size for multi-GPU
NGPUS=$(($SLURM_NNODES * 8))
export GBS=8192

### Network / RCCL settings ###
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0

### Proxy for wandb ###
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

### MIOpen settings ###
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

### Output dirs ###
mkdir -p logs/ckpt_ids logs/run_ids checkpoints data/results

SINN_CKPT_DIR="checkpoints/sinn-${SLURM_JOB_ID}"
mkdir -p "$SINN_CKPT_DIR"
echo "[SINN] Checkpoint dir: $SINN_CKPT_DIR"

# Track checkpoint paths across phases
G_CKPT_PATH=""
E_CKPT_PATH=""
GE_CKPT_PATH=""

### ═══════════════════════════════════════════════════════════════
### Phase G: Train genotype encoder on G_hat
### ═══════════════════════════════════════════════════════════════
if [[ "$RUN_PHASE_G" == "True" ]]; then
    echo ""
    echo "============================================"
    echo "[SINN] Phase G: Training genotype encoder"
    echo "============================================"

    export WANDB_RUN_ID_FILE="logs/run_ids/wandb_sinn_g_${SLURM_JOB_ID}.txt"
    export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_sinn_g_${SLURM_JOB_ID}.txt"
    rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"

    srun -N ${SLURM_NNODES} \
        -n ${NGPUS} \
        --gpus-per-task=1 \
        --cpu-bind=cores \
        --export=ALL \
        --kill-on-bad-exit=1 \
        python -u scripts/train_sinn.py \
            --sinn_phase g \
            --decomp_path "$DECOMP_PATH" \
            --batch_size $G_BATCH_SIZE \
            --gbs $GBS \
            --lr $G_LR \
            --num_epochs $G_EPOCHS \
            --early_stop $G_EARLY_STOP \
            --g_input_type $G_INPUT_TYPE \
            --env_categorical_mode $ENV_CATEGORICAL_MODE \
            --emb_size $EMB_SIZE \
            --heads $HEADS \
            --g_layers $G_LAYERS \
            --mlp_layers $MLP_LAYERS \
            --gxe_layers $GXE_LAYERS \
            --dropout $DROPOUT \
            --seed $SEED \
            --scale_targets $SCALE_TARGETS \
            --leo_val $LEO_VAL \
            --leo_val_fraction $LEO_VAL_FRACTION

    # Find the checkpoint
    if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
        G_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
        G_CKPT_PATH="${G_CKPT_DIR}/best_g.pt"
        echo "[SINN] Phase G checkpoint: $G_CKPT_PATH"
        cp "$G_CKPT_PATH" "${SINN_CKPT_DIR}/best_g.pt" 2>/dev/null || true
    else
        echo "[WARN] Phase G checkpoint dir not found"
    fi
fi

### ═══════════════════════════════════════════════════════════════
### Phase E: Train environment encoder on E_hat
### ═══════════════════════════════════════════════════════════════
if [[ "$RUN_PHASE_E" == "True" ]]; then
    echo ""
    echo "============================================"
    echo "[SINN] Phase E: Training environment encoder"
    echo "============================================"

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
            --gxe_layers $GXE_LAYERS \
            --dropout $DROPOUT \
            --seed $SEED \
            --scale_targets $SCALE_TARGETS \
            --leo_val $LEO_VAL \
            --leo_val_fraction $LEO_VAL_FRACTION

    if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
        E_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
        E_CKPT_PATH="${E_CKPT_DIR}/best_e.pt"
        echo "[SINN] Phase E checkpoint: $E_CKPT_PATH"
        cp "$E_CKPT_PATH" "${SINN_CKPT_DIR}/best_e.pt" 2>/dev/null || true
    else
        echo "[WARN] Phase E checkpoint dir not found"
    fi
fi

### ═══════════════════════════════════════════════════════════════
### Phase GE: Train interaction on GE_hat residual
### ═══════════════════════════════════════════════════════════════
if [[ "$RUN_PHASE_GE" == "True" ]]; then
    echo ""
    echo "============================================"
    echo "[SINN] Phase GE: Training interaction model"
    echo "============================================"

    GE_EXTRA_ARGS=""
    if [[ -n "$G_CKPT_PATH" && -f "$G_CKPT_PATH" ]]; then
        GE_EXTRA_ARGS="$GE_EXTRA_ARGS --g_ckpt $G_CKPT_PATH"
    fi
    if [[ -n "$E_CKPT_PATH" && -f "$E_CKPT_PATH" ]]; then
        GE_EXTRA_ARGS="$GE_EXTRA_ARGS --e_ckpt $E_CKPT_PATH"
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
            --gxe_layers $GXE_LAYERS \
            --dropout $DROPOUT \
            --seed $SEED \
            --scale_targets $SCALE_TARGETS \
            --leo_val $LEO_VAL \
            --leo_val_fraction $LEO_VAL_FRACTION \
            --freeze_encoders $FREEZE_ENCODERS \
            $GE_EXTRA_ARGS

    if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
        GE_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
        GE_CKPT_PATH="${GE_CKPT_DIR}/best_ge.pt"
        echo "[SINN] Phase GE checkpoint: $GE_CKPT_PATH"
        cp "$GE_CKPT_PATH" "${SINN_CKPT_DIR}/best_ge.pt" 2>/dev/null || true
    else
        echo "[WARN] Phase GE checkpoint dir not found"
    fi
fi

### ═══════════════════════════════════════════════════════════════
### Phase Finetune: End-to-end on raw yield with envpcc loss
### ═══════════════════════════════════════════════════════════════
if [[ "$RUN_PHASE_FINETUNE" == "True" ]]; then
    echo ""
    echo "============================================"
    echo "[SINN] Phase Finetune: End-to-end fine-tuning"
    echo "============================================"

    FT_EXTRA_ARGS=""
    if [[ -n "$GE_CKPT_PATH" && -f "$GE_CKPT_PATH" ]]; then
        FT_EXTRA_ARGS="--ge_ckpt $GE_CKPT_PATH"
    fi

    # Compute microbatch for finetune (may use more nodes)
    FT_MBS=$(($GBS / $NGPUS))

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
            --batch_size $FT_MBS \
            --gbs $GBS \
            --lr $FT_LR \
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
            --gxe_layers $GXE_LAYERS \
            --dropout $DROPOUT \
            --seed $SEED \
            --scale_targets $SCALE_TARGETS \
            --leo_val $LEO_VAL \
            --leo_val_fraction $LEO_VAL_FRACTION \
            $FT_EXTRA_ARGS

    if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
        FT_CKPT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
        FT_CKPT_PATH="${FT_CKPT_DIR}/best_finetune.pt"
        echo "[SINN] Phase Finetune checkpoint: $FT_CKPT_PATH"
        cp "$FT_CKPT_PATH" "${SINN_CKPT_DIR}/best_finetune.pt" 2>/dev/null || true
    else
        echo "[WARN] Phase Finetune checkpoint dir not found"
    fi

    ### Eval on test set ###
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
fi

echo ""
echo "============================================"
echo "[SINN] All phases complete."
echo "[SINN] Checkpoints saved to: $SINN_CKPT_DIR"
echo "============================================"

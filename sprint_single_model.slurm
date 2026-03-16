#!/bin/bash
# Focused single-model sprint launcher.
# Usage examples:
#   sbatch --export=ALL,VARIANT=baseline,STAGE=A sprint_single_model.slurm
#   sbatch --export=ALL,VARIANT=affine,STAGE=A sprint_single_model.slurm
#   sbatch --export=ALL,VARIANT=affine,STAGE=B,RESUME_CHECKPOINT=/path/to/checkpoint.pt sprint_single_model.slurm
#   sbatch --export=ALL,VARIANT=affine_ccc,STAGE=C,RESUME_CHECKPOINT=/path/to/checkpoint.pt sprint_single_model.slurm
#   sbatch --export=ALL,VARIANT=bilinear,STAGE=C,RESUME_CHECKPOINT=/path/to/checkpoint.pt sprint_single_model.slurm
# Variants:
#   baseline   -> plain FullTransformer baseline on proxy split
#   affine     -> affine residual model with linear residual head
#   affine_ccc -> affine residual model; stage C uses envccc
#   bilinear   -> affine residual model with bilinear residual head

#SBATCH -A SYB114
#SBATCH -J sprint1
#SBATCH -N 4
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -eo pipefail

ulimit -S -c 0

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH=/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH}

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

export VARIANT="${VARIANT:-baseline}"
export STAGE="${STAGE:-A}"

case "$VARIANT" in
  baseline|affine|affine_ccc|bilinear) ;;
  *)
    echo "[ERROR] Unsupported VARIANT=$VARIANT"
    exit 2
    ;;
esac

case "$STAGE" in
  A|B|C) ;;
  *)
    echo "[ERROR] Unsupported STAGE=$STAGE"
    exit 2
    ;;
esac

export G_ENC=True
export E_ENC=True
export LD_ENC=True
export GXE_ENC=tf
export WG=True
export FULL_TRANSFORMER=True
export G_ENCODER_TYPE=moe
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=2
export MOE_SHARED_EXPERT=True
export MOE_EXPERT_HIDDEN_DIM=256
export MOE_SHARED_EXPERT_HIDDEN_DIM=256
export MOE_LOSS_WEIGHT=0.01
export FULL_TF_MLP_TYPE=moe

export GBS=8192
NGPUS=$(($SLURM_NNODES * 8))
MBS=$(($GBS / $NGPUS))
export BATCH_SIZE=$MBS
echo "Training with global batch size $GBS on $NGPUS GPUs (microbatch size $MBS)"

export G_LAYERS=1
export LD_LAYERS=1
export MLP_LAYERS=1
export GXE_LAYERS=1
export HEADS=4
export EMB_SIZE=256
export DROPOUT=0.15
export WEIGHT_DECAY=1e-5
export SEED=1
export SCALE_TARGETS=False
export G_INPUT_TYPE=tokens
export ENV_CATEGORICAL_MODE=drop
export ENV_STRATIFIED=True
export MIN_SAMPLES_PER_ENV=32

export CONTRASTIVE_MODE=none
export CONTRASTIVE_WEIGHT=0.1
export CONTRASTIVE_TEMPERATURE=0.1
export CONTRASTIVE_SIM_TYPE=grm
export CONTRASTIVE_LOSS_TYPE=mse
export ENV_CONTRASTIVE_WEIGHT=0.1
export ENV_CONTRASTIVE_TEMPERATURE=0.5

export VAL_SCHEME=proxy_same_tester
export PROXY_TESTER=PHP02
export PROXY_HOLDOUT_FRAC=0.25
export PROXY_SEED=1
export LEO_VAL=False
export LEO_VAL_FRACTION=0.10

export RESIDUAL=False
export RESIDUAL_MODE=affine
export RESIDUAL_HEAD_TYPE=linear
export BILINEAR_RANK=32
export DETACH_YMEAN=False
export LAMBDA_YMEAN=0.0
export LAMBDA_RESID=0.0
export FREEZE_BACKBONE=False
export FREEZE_RANK_HEAD=False
export UNFREEZE_LAST_BLOCK_ONLY=False
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
export BACKBONE_LR="${BACKBONE_LR:-}"
export HEAD_LR="${HEAD_LR:-}"

if [[ "$VARIANT" != "baseline" ]]; then
  export RESIDUAL=True
fi
if [[ "$VARIANT" == "bilinear" ]]; then
  export RESIDUAL_HEAD_TYPE=bilinear
fi

case "$STAGE" in
  A)
    export NUM_EPOCHS=3000
    export EARLY_STOP=200
    export LR=1e-4
    export LOSS=envpcc
    export LOSS_WEIGHTS=1.0
    ;;
  B)
    if [[ -z "$RESUME_CHECKPOINT" ]]; then
      echo "[ERROR] STAGE=B requires RESUME_CHECKPOINT"
      exit 2
    fi
    if [[ "$VARIANT" == "baseline" ]]; then
      echo "[ERROR] STAGE=B is only valid for residual variants"
      exit 2
    fi
    export NUM_EPOCHS=200
    export EARLY_STOP=40
    export LR=5e-5
    export LOSS=mse
    export LOSS_WEIGHTS=1.0
    export FREEZE_BACKBONE=True
    export FREEZE_RANK_HEAD=True
    export HEAD_LR=5e-5
    ;;
  C)
    if [[ -z "$RESUME_CHECKPOINT" ]]; then
      echo "[ERROR] STAGE=C requires RESUME_CHECKPOINT"
      exit 2
    fi
    if [[ "$VARIANT" == "baseline" ]]; then
      echo "[ERROR] STAGE=C is only valid for residual variants"
      exit 2
    fi
    export NUM_EPOCHS=100
    export EARLY_STOP=20
    export LR=1e-5
    export BACKBONE_LR=1e-5
    export HEAD_LR=5e-5
    export UNFREEZE_LAST_BLOCK_ONLY=True
    if [[ "$VARIANT" == "affine_ccc" ]]; then
      export LOSS=envccc
      export LOSS_WEIGHTS=1.0
    else
      export LOSS=envpcc+mse
      export LOSS_WEIGHTS=1.0,0.05
    fi
    ;;
esac

export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

mkdir -p logs/ckpt_ids logs/run_ids checkpoints data/results
export WANDB_RUN_ID_FILE="logs/run_ids/wandb_run_id_${SLURM_JOB_ID}.txt"
export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_dir_${SLURM_JOB_ID}.txt"
rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0

export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

TRAIN_SCRIPT="scripts/train.py"
if [[ "$VARIANT" != "baseline" ]]; then
  TRAIN_SCRIPT="scripts/train_residual.py"
fi

NGPUS=$(($SLURM_NNODES * 8))
srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u "$TRAIN_SCRIPT" \
        --batch_size $BATCH_SIZE \
        --gbs $GBS \
        --g_enc $G_ENC \
        --e_enc $E_ENC \
        --ld_enc $LD_ENC \
        --gxe_enc $GXE_ENC \
        --wg $WG \
        --full_transformer $FULL_TRANSFORMER \
        --g_encoder_type $G_ENCODER_TYPE \
        --moe_num_experts $MOE_NUM_EXPERTS \
        --moe_top_k $MOE_TOP_K \
        --moe_shared_expert $MOE_SHARED_EXPERT \
        --moe_expert_hidden_dim $MOE_EXPERT_HIDDEN_DIM \
        --moe_shared_expert_hidden_dim $MOE_SHARED_EXPERT_HIDDEN_DIM \
        --moe_loss_weight $MOE_LOSS_WEIGHT \
        --full_tf_mlp_type $FULL_TF_MLP_TYPE \
        --num_epochs $NUM_EPOCHS \
        --lr $LR \
        --backbone_lr "${BACKBONE_LR:-$LR}" \
        --head_lr "${HEAD_LR:-$LR}" \
        --weight_decay $WEIGHT_DECAY \
        --dropout $DROPOUT \
        --early_stop $EARLY_STOP \
        --g_layers $G_LAYERS \
        --ld_layers $LD_LAYERS \
        --mlp_layers $MLP_LAYERS \
        --gxe_layers $GXE_LAYERS \
        --heads $HEADS \
        --emb_size $EMB_SIZE \
        --seed $SEED \
        --g_input_type $G_INPUT_TYPE \
        --env_categorical_mode $ENV_CATEGORICAL_MODE \
        --loss $LOSS \
        --loss_weights $LOSS_WEIGHTS \
        --scale_targets $SCALE_TARGETS \
        --env_stratified $ENV_STRATIFIED \
        --min_samples_per_env $MIN_SAMPLES_PER_ENV \
        --contrastive_mode $CONTRASTIVE_MODE \
        --contrastive_weight $CONTRASTIVE_WEIGHT \
        --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
        --contrastive_sim_type $CONTRASTIVE_SIM_TYPE \
        --contrastive_loss_type $CONTRASTIVE_LOSS_TYPE \
        --env_contrastive_weight $ENV_CONTRASTIVE_WEIGHT \
        --env_contrastive_temperature $ENV_CONTRASTIVE_TEMPERATURE \
        --leo_val $LEO_VAL \
        --leo_val_fraction $LEO_VAL_FRACTION \
        --val_scheme $VAL_SCHEME \
        --proxy_tester $PROXY_TESTER \
        --proxy_holdout_frac $PROXY_HOLDOUT_FRAC \
        --proxy_seed $PROXY_SEED \
        --residual $RESIDUAL \
        --residual_mode $RESIDUAL_MODE \
        --residual_head_type $RESIDUAL_HEAD_TYPE \
        --bilinear_rank $BILINEAR_RANK \
        --detach_ymean $DETACH_YMEAN \
        --lambda_ymean $LAMBDA_YMEAN \
        --lambda_resid $LAMBDA_RESID \
        --resume_checkpoint "$RESUME_CHECKPOINT" \
        --freeze_backbone $FREEZE_BACKBONE \
        --freeze_rank_head $FREEZE_RANK_HEAD \
        --unfreeze_last_block_only $UNFREEZE_LAST_BLOCK_ONLY

if [[ -f "$WANDB_RUN_ID_FILE" ]]; then
    export WANDB_RESUME=allow
    export WANDB_RUN_ID=$(cat "$WANDB_RUN_ID_FILE")
else
    echo "[WARNING] WandB run ID file not found; eval will create a new run."
    unset WANDB_RESUME
    unset WANDB_RUN_ID
fi

if [[ -f "$CHECKPOINT_DIR_FILE" ]]; then
    CHECKPOINT_DIR=$(cat "$CHECKPOINT_DIR_FILE")
    echo "[INFO] Using checkpoint dir from train: $CHECKPOINT_DIR"
else
    echo "[ERROR] checkpoint dir file not found: $CHECKPOINT_DIR_FILE"
    exit 2
fi

python -u scripts/eval.py \
    --batch_size $BATCH_SIZE \
    --gbs $GBS \
    --g_enc $G_ENC \
    --e_enc $E_ENC \
    --ld_enc $LD_ENC \
    --gxe_enc $GXE_ENC \
    --wg $WG \
    --full_transformer $FULL_TRANSFORMER \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --early_stop $EARLY_STOP \
    --g_layers $G_LAYERS \
    --ld_layers $LD_LAYERS \
    --mlp_layers $MLP_LAYERS \
    --gxe_layers $GXE_LAYERS \
    --heads $HEADS \
    --emb_size $EMB_SIZE \
    --g_input_type $G_INPUT_TYPE \
    --env_categorical_mode $ENV_CATEGORICAL_MODE \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --loss $LOSS \
    --loss_weights $LOSS_WEIGHTS \
    --scale_targets $SCALE_TARGETS \
    --residual $RESIDUAL \
    --residual_mode $RESIDUAL_MODE \
    --residual_head_type $RESIDUAL_HEAD_TYPE \
    --bilinear_rank $BILINEAR_RANK

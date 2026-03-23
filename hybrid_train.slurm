#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J hybrid-joint-fulltf
#SBATCH -N 4
#SBATCH -t 02:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -eo pipefail

ulimit -S -c 0

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

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

export GBS=8192
NGPUS=$(($SLURM_NNODES * 8))
MBS=$(($GBS / $NGPUS))
export BATCH_SIZE=$MBS
echo "Training with global batch size $GBS on $NGPUS GPUs (microbatch size $MBS)"

export NUM_EPOCHS=3000
export LR=1e-4
export WEIGHT_DECAY=1e-5
export DROPOUT=0.15
export EARLY_STOP=200

export G_LAYERS=1
export LD_LAYERS=1
export MLP_LAYERS=1
export GXE_LAYERS=1

export HEADS=4
export EMB_SIZE=256
export SEED=${SEED:-1}
export MODEL_SEED=${MODEL_SEED:-$SEED}
export SPLIT_SEED=${SPLIT_SEED:-$SEED}
export SCALE_TARGETS=False
export G_INPUT_TYPE=tokens
export ENV_CATEGORICAL_MODE=drop

export LOSS="envpcc"
export LOSS_WEIGHTS="1.0"
export ENV_STRATIFIED=True
export MIN_SAMPLES_PER_ENV=32

export CONTRASTIVE_MODE=e
export CONTRASTIVE_WEIGHT=0.1
export CONTRASTIVE_TEMPERATURE=0.1
export CONTRASTIVE_SIM_TYPE=grm
export CONTRASTIVE_LOSS_TYPE=mse
export ENV_CONTRASTIVE_WEIGHT=0.1
export ENV_CONTRASTIVE_TEMPERATURE=0.5
export CONTRASTIVE_WARMUP_EPOCHS=50
export CONTRASTIVE_RAMP_EPOCHS=50

export VAL_SCHEME=hybrid_combo
export LEO_VAL=True
export LEO_VAL_FRACTION=0.15
export PROXY_TESTER=PHP02
export PROXY_HOLDOUT_FRAC=${PROXY_HOLDOUT_FRAC:-0.10}
export PROXY_SEED=${PROXY_SEED:-1}
export PROXY_DISJOINT_FROM_LEO=True
export PROXY_SCORE_WEIGHT=0.55
export LEO_SCORE_WEIGHT=0.25
export SCALE_SCORE_WEIGHT=0.20
export CHECKPOINT_METRIC=leo

export CALIBRATION_MODE=env_affine
export RANK_AUX_LOSS=none
export RANK_AUX_WEIGHT=0.05
export RANK_AUX_START_EPOCH=100
export CALIBRATION_START_EPOCH=150
export CALIBRATION_RAMP_EPOCHS=100
export CALIBRATION_DETACH_RANK_UNTIL_EPOCH=250
export CALIBRATION_JOINT_GRAD_FRACTION=0.10
export ENVCCC_WEIGHT=${ENVCCC_WEIGHT:-0.10}
export HUBER_WEIGHT=${HUBER_WEIGHT:-0.02}
export HUBER_DELTA=1.0

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

srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train.py \
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
        --seed $SEED \
        --model_seed $MODEL_SEED \
        --split_seed $SPLIT_SEED \
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
        --contrastive_warmup_epochs $CONTRASTIVE_WARMUP_EPOCHS \
        --contrastive_ramp_epochs $CONTRASTIVE_RAMP_EPOCHS \
        --val_scheme $VAL_SCHEME \
        --leo_val $LEO_VAL \
        --leo_val_fraction $LEO_VAL_FRACTION \
        --proxy_tester $PROXY_TESTER \
        --proxy_holdout_frac $PROXY_HOLDOUT_FRAC \
        --proxy_seed $PROXY_SEED \
        --proxy_disjoint_from_leo $PROXY_DISJOINT_FROM_LEO \
        --proxy_score_weight $PROXY_SCORE_WEIGHT \
        --leo_score_weight $LEO_SCORE_WEIGHT \
        --scale_score_weight $SCALE_SCORE_WEIGHT \
        --checkpoint_metric $CHECKPOINT_METRIC \
        --calibration_mode $CALIBRATION_MODE \
        --rank_aux_loss $RANK_AUX_LOSS \
        --rank_aux_weight $RANK_AUX_WEIGHT \
        --rank_aux_start_epoch $RANK_AUX_START_EPOCH \
        --calibration_start_epoch $CALIBRATION_START_EPOCH \
        --calibration_ramp_epochs $CALIBRATION_RAMP_EPOCHS \
        --calibration_detach_rank_until_epoch $CALIBRATION_DETACH_RANK_UNTIL_EPOCH \
        --calibration_joint_grad_fraction $CALIBRATION_JOINT_GRAD_FRACTION \
        --envccc_weight $ENVCCC_WEIGHT \
        --huber_weight $HUBER_WEIGHT \
        --huber_delta $HUBER_DELTA

if [[ -f "$WANDB_RUN_ID_FILE" ]]; then
    export WANDB_RESUME=allow
    export WANDB_RUN_ID=$(cat "$WANDB_RUN_ID_FILE")
else
    echo "[WARNING] WandB run ID file not found... Eval will create a new run."
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

run_eval() {
    local checkpoint_tag="$1"
    local metric_prefix="$2"
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
        --seed $SEED \
        --model_seed $MODEL_SEED \
        --split_seed $SPLIT_SEED \
        --g_input_type $G_INPUT_TYPE \
        --env_categorical_mode $ENV_CATEGORICAL_MODE \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --checkpoint_tag "$checkpoint_tag" \
        --metric_prefix "$metric_prefix" \
        --loss $LOSS \
        --loss_weights $LOSS_WEIGHTS \
        --scale_targets $SCALE_TARGETS \
        --calibration_mode $CALIBRATION_MODE
}

run_eval best_leo test
if [[ -f "$CHECKPOINT_DIR/best_select.pt" ]]; then
    run_eval best_select test_select
fi
if [[ -f "$CHECKPOINT_DIR/best_scale.pt" ]]; then
    run_eval best_scale test_scale
fi
run_eval latest test_latest

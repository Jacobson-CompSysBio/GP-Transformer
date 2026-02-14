#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J debug-rolling-firsttest
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
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

### first rolling test: best-model-like hyperparams, recent folds only ###
export G_ENC=True
export E_ENC=True
export LD_ENC=True
export GXE_ENC=tf
export WG=True
export FULL_TRANSFORMER=True
export RES=False

export G_ENCODER_TYPE=moe
export FULL_TF_MLP_TYPE=moe
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=2
export MOE_SHARED_EXPERT=True
export MOE_EXPERT_HIDDEN_DIM=256
export MOE_SHARED_EXPERT_HIDDEN_DIM=256
export MOE_LOSS_WEIGHT=0.01

# smaller budget for debug queue while preserving optimization dynamics
export GBS=2048
NGPUS=$(($SLURM_NNODES * 8))
MBS=$(($GBS / $NGPUS))
export BATCH_SIZE=$MBS
echo "Debug rolling test with global batch size $GBS on $NGPUS GPUs (microbatch size $MBS)"

export NUM_EPOCHS=10
export LR=1e-4
export WEIGHT_DECAY=1e-5
export DROPOUT=0.15
export EARLY_STOP=10

export G_LAYERS=1
export LD_LAYERS=1
export MLP_LAYERS=1
export GXE_LAYERS=1

export HEADS=4
export EMB_SIZE=256
export SEED=1
export SCALE_TARGETS=True
export G_INPUT_TYPE=tokens

export LOSS="envpcc"
export LOSS_WEIGHTS="1.0"

# these only apply when resid is true
export DETACH_YMEAN=True
export LAMBDA_YMEAN=0.5
export LAMBDA_RESID=1.0

export ENV_STRATIFIED=True
export MIN_SAMPLES_PER_ENV=32

export CONTRASTIVE_MODE=none
export CONTRASTIVE_WEIGHT=0.1
export CONTRASTIVE_TEMPERATURE=0.1
export CONTRASTIVE_SIM_TYPE=grm
export CONTRASTIVE_LOSS_TYPE=mse
export ENV_CONTRASTIVE_WEIGHT=0.1
export ENV_CONTRASTIVE_TEMPERATURE=0.5

# recent folds only for a tractable first comparison
export ROLLING_VAL_YEARS="2021,2022,2023"
export ROLLING_MAX_FOLDS=3
export ROLLING_RECENT_FIRST=True

### wandb settings ###
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

mkdir -p logs/ckpt_ids logs/run_ids checkpoints data/results
export WANDB_RUN_ID_FILE="logs/run_ids/wandb_run_id_${SLURM_JOB_ID}.txt"
export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_dir_${SLURM_JOB_ID}.txt"
rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"

### network / rccl settings ###
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0

### miopen settings ###
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

### srun ###
srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train_rolling.py \
        --batch_size $BATCH_SIZE \
        --gbs $GBS \
        --g_enc $G_ENC \
        --e_enc $E_ENC \
        --ld_enc $LD_ENC \
        --gxe_enc $GXE_ENC \
        --wg $WG \
        --full_transformer $FULL_TRANSFORMER \
        --residual $RES \
        --g_encoder_type $G_ENCODER_TYPE \
        --full_tf_mlp_type $FULL_TF_MLP_TYPE \
        --moe_num_experts $MOE_NUM_EXPERTS \
        --moe_top_k $MOE_TOP_K \
        --moe_shared_expert $MOE_SHARED_EXPERT \
        --moe_expert_hidden_dim $MOE_EXPERT_HIDDEN_DIM \
        --moe_shared_expert_hidden_dim $MOE_SHARED_EXPERT_HIDDEN_DIM \
        --moe_loss_weight $MOE_LOSS_WEIGHT \
        --detach_ymean $DETACH_YMEAN \
        --lambda_ymean $LAMBDA_YMEAN \
        --lambda_resid $LAMBDA_RESID \
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
        --loss $LOSS \
        --loss_weights $LOSS_WEIGHTS \
        --scale_targets $SCALE_TARGETS \
        --g_input_type $G_INPUT_TYPE \
        --env_stratified $ENV_STRATIFIED \
        --min_samples_per_env $MIN_SAMPLES_PER_ENV \
        --contrastive_mode $CONTRASTIVE_MODE \
        --contrastive_weight $CONTRASTIVE_WEIGHT \
        --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
        --contrastive_sim_type $CONTRASTIVE_SIM_TYPE \
        --contrastive_loss_type $CONTRASTIVE_LOSS_TYPE \
        --env_contrastive_weight $ENV_CONTRASTIVE_WEIGHT \
        --env_contrastive_temperature $ENV_CONTRASTIVE_TEMPERATURE

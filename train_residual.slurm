#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J 3p,strat,moe,res
#SBATCH -N 8
#SBATCH -t 02:00:00
#SBATCH -p batch 
#SBATCH -o logs/%x-%j.out # Out Path
#SBATCH -e logs/%x-%j.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

### env, modules, and settings ###
set -eo pipefail

ulimit -S -c 0 # disaple core dumps

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

### train hyperparameters ###
export G_ENC=True
export E_ENC=True
export LD_ENC=True
export GXE_ENC=tf  # options: "tf", "mlp", "cnn"
export WG=True  # weighted gate for 3-prong architecture
export RES=True
export FULL_TRANSFORMER=True
export G_ENCODER_TYPE=moe  # options: "dense", "moe"
export MOE_NUM_EXPERTS=4
export MOE_TOP_K=2
export MOE_SHARED_EXPERT=True # set to True to enable shared expert
export MOE_LOSS_WEIGHT=0.01

# calculate microbatch size given global batch size of 2048 and N nodes * 8 gpus/node = 64 gpus
export GBS=16384
NGPUS=$(($SLURM_NNODES * 8))
MBS=$(($GBS / $NGPUS))
export BATCH_SIZE=$MBS
echo "Training with global batch size $GBS on $NGPUS GPUs (microbatch size $MBS)"

export NUM_EPOCHS=3000
export LR=1e-4
export WEIGHT_DECAY=1e-4
export DROPOUT=0.25
export EARLY_STOP=200

export G_LAYERS=1
export LD_LAYERS=1
export MLP_LAYERS=1
export GXE_LAYERS=1

export HEADS=4
export EMB_SIZE=256
export MOE_EXPERT_HIDDEN_DIM=$EMB_SIZE    # override default (4x) to match emb dim
export MOE_SHARED_EXPERT_HIDDEN_DIM=$EMB_SIZE
export LOSS="envpcc"
export LOSS_WEIGHTS="1.0"
export SCALE_TARGETS=False
export ENV_STRATIFIED=True  # use environment-stratified batching for envpcc loss
export MIN_SAMPLES_PER_ENV=32  # samples per environment per batch for stable correlation

# Contrastive loss: encourages G embeddings to reflect genetic similarity
# This can improve generalization by learning meaningful genomic representations
# NOTE: Now uses warmup - doesn't activate until epoch 50
export CONTRASTIVE_LOSS=True        # Set to True to enable
export CONTRASTIVE_WEIGHT=0.05      # Lower weight (was 0.1) - ramps up after warmup
export CONTRASTIVE_TEMPERATURE=0.5  # temperature for softmax (higher = softer)
export CONTRASTIVE_SIM_TYPE=grm     # 'grm' (recommended) or 'ibs'
export CONTRASTIVE_LOSS_TYPE=mse    # 'mse' (recommended), 'cosine', or 'kl'

# LEO (Leave-Environment-Out) validation: hold out entire environments for val
# This better simulates test performance on unseen environments
export LEO_VAL=True
export LEO_VAL_FRACTION=0.15  # fraction of environments to hold out (default 15%)

### wandb settings ###
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

# make wandb logging dir
mkdir -p logs/ckpt_ids logs/run_ids checkpoints data/results
export WANDB_RUN_ID_FILE="logs/run_ids/wandb_run_id_${SLURM_JOB_ID}.txt"
export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_dir_${SLURM_JOB_ID}.txt"
rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"

### network / rccl settings ###
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0

### miopen settings ###
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

### srun (model train) ###
NGPUS=$(($SLURM_NNODES * 8))
srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train_residual.py \
        --batch_size $BATCH_SIZE \
        --gbs $GBS \
        --g_enc $G_ENC \
        --e_enc $E_ENC \
        --ld_enc $LD_ENC \
        --gxe_enc $GXE_ENC \
        --wg $WG \
        --full_transformer $FULL_TRANSFORMER \
        --residual $RES \
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
        --loss $LOSS \
        --loss_weights $LOSS_WEIGHTS \
        --scale_targets $SCALE_TARGETS \
        --env_stratified $ENV_STRATIFIED \
        --min_samples_per_env $MIN_SAMPLES_PER_ENV \
        --contrastive_loss $CONTRASTIVE_LOSS \
        --contrastive_weight $CONTRASTIVE_WEIGHT \
        --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
        --contrastive_sim_type $CONTRASTIVE_SIM_TYPE \
        --contrastive_loss_type $CONTRASTIVE_LOSS_TYPE \
        --leo_val $LEO_VAL \
        --leo_val_fraction $LEO_VAL_FRACTION

### python (model eval) ###
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

python -u scripts/eval.py \
    --batch_size $BATCH_SIZE \
    --gbs $GBS \
    --g_enc $G_ENC \
    --e_enc $E_ENC \
    --ld_enc $LD_ENC \
    --gxe_enc $GXE_ENC \
    --wg $WG \
    --full_transformer $FULL_TRANSFORMER \
    --residual $RES \
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
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --loss $LOSS \
    --loss_weights $LOSS_WEIGHTS \
    --scale_targets $SCALE_TARGETS

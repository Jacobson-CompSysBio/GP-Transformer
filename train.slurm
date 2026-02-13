#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J baseline
#SBATCH -N 4
#SBATCH -t 2:00:00
#SBATCH -p batch
##SBATCH -q debug
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
export FULL_TRANSFORMER=True  # use FullTransformer architecture

# MoE settings (used when G_ENCODER_TYPE=moe or FULL_TF_MLP_TYPE=moe)
export G_ENCODER_TYPE=moe  # options: "dense", "moe"
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=2
export MOE_SHARED_EXPERT=True
export MOE_EXPERT_HIDDEN_DIM=256
export MOE_SHARED_EXPERT_HIDDEN_DIM=256
export MOE_LOSS_WEIGHT=0.01

# calculate microbatch size given global batch size of 8192 and N nodes * 8 gpus/node = 32 gpus
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
export GXE_LAYERS=1 # applies to FullTF if used

export HEADS=4
export EMB_SIZE=256
export SEED=1
export SCALE_TARGETS=False
export G_INPUT_TYPE=tokens    # options: "tokens", "grm"
export ENV_FEATURE_ID_EMB=True
export ENV_STAGE_ID_EMB=True
export ENV_CAT_EMBEDDINGS=True

export LOSS="envpcc" # + separated list of losses
export LOSS_WEIGHTS="1.0" # comma separated list of weights for each loss
export ENV_STRATIFIED=True  # use environment-stratified batching for envpcc loss
export MIN_SAMPLES_PER_ENV=32  # samples per environment per batch for stable correlation

# Contrastive loss
export CONTRASTIVE_MODE=none  # ablation: none, g, e, g+e
export CONTRASTIVE_WEIGHT=0.1
export CONTRASTIVE_TEMPERATURE=0.1
export CONTRASTIVE_SIM_TYPE=grm      # options: grm, ibs
export CONTRASTIVE_LOSS_TYPE=mse     # options: mse, cosine, kl
export ENV_CONTRASTIVE_WEIGHT=0.1
export ENV_CONTRASTIVE_TEMPERATURE=0.5

# LEO (Leave-Environment-Out) validation
export LEO_VAL=True
export LEO_VAL_FRACTION=0.15

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
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
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
        --env_feature_id_emb $ENV_FEATURE_ID_EMB \
        --env_stage_id_emb $ENV_STAGE_ID_EMB \
        --env_cat_embeddings $ENV_CAT_EMBEDDINGS \
        --seed $SEED \
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
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --loss $LOSS \
    --loss_weights $LOSS_WEIGHTS \
    --scale_targets $SCALE_TARGETS

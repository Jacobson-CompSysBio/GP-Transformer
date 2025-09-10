#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J gxe-transformer
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
export MOE=True

export BATCH_SIZE=256
export NUM_EPOCHS=5000
export LR=1e-4
export WEIGHT_DECAY=1e-5
export DROPOUT=0.5

export G_LAYERS=1
export LD_LAYERS=2
export MLP_LAYERS=1
export GXE_LAYERS=2

export HEADS=4
export EMB_SIZE=256
export LOSS="pcc"
export ALPHA=0.0001 # only used if LOSS is "both"

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
    python -u scripts/train.py \
        --batch_size $BATCH_SIZE \
        --g_enc $G_ENC \
        --e_enc $E_ENC \
        --ld_enc $LD_ENC \
        --gxe_enc $GXE_ENC \
        --moe $MOE \
        --num_epochs $NUM_EPOCHS \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY \
        --dropout $DROPOUT \
        --g_layers $G_LAYERS \
        --ld_layers $LD_LAYERS \
        --mlp_layers $MLP_LAYERS \
        --gxe_layers $GXE_LAYERS \
        --heads $HEADS \
        --emb_size $EMB_SIZE \
        --loss $LOSS \
        --alpha $ALPHA

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
    --g_enc $G_ENC \
    --e_enc $E_ENC \
    --ld_enc $LD_ENC \
    --gxe_enc $GXE_ENC \
    --moe $MOE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --g_layers $G_LAYERS \
    --ld_layers $LD_LAYERS \
    --mlp_layers $MLP_LAYERS \
    --gxe_layers $GXE_LAYERS \
    --heads $HEADS \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --loss $LOSS \
    --alpha $ALPHA
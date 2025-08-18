#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J gxe-tf-ddp 
#SBATCH -N 48 
#SBATCH -t 8:00:00
#SBATCH -p extended 
#SBATCH -o logs/%x.out # Out Path
#SBATCH -e logs/%x.err # Err Path
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

### wandb settings ###
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

### network / rccl settings ###
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0

### miopen settings ###
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

### srun ###
NGPUS=$(($SLURM_NNODES * 8))
srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train.py --batch_size 64 \
        --model_type ft \
        --num_epochs 100_000 \
        --lr 1e-4 \
        --layers_per_block 4 \
        --heads 16 \
        --emb_size 768
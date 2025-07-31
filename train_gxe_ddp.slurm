#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J gxe-tf-ddp 
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH -p batch 
#SBATCH -q debug
#SBATCH -C nvme
#SBATCH --gres=gpu:8
#SBATCH -o logs/%x.out # Out Path
#SBATCH -e logs/%x.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

set -eo pipefail

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

srun python -u scripts/train.py \
    --batch_size 32
    --num_iters 50000

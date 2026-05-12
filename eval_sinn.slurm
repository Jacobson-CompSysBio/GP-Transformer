#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J sinn-eval
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o logs/sinn-eval-%j.out
#SBATCH -e logs/sinn-eval-%j.err
#SBATCH --open-mode=truncate

set -eo pipefail

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy

cd /lustre/orion/syb111/proj-shared/Personal/krusepi/projects/phenotyping/GP-Transformer

CKPT_DIR="checkpoints/sinn-4193293"

echo "Evaluating SINN checkpoint: $CKPT_DIR"
python -u scripts/eval.py --checkpoint_dir "$CKPT_DIR"
echo "Done."

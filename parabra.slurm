#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J parabra
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=56
#SBATCH --threads-per-core=1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

module purge

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
DATA_ROOT=${DATA_ROOT:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/phenotyping/GP-Transformer/data}
PYTHON_BIN=${PYTHON_BIN:-/lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin/python}
TRANSFORMER_PREDICTIONS=${TRANSFORMER_PREDICTIONS:-${DATA_ROOT}/results/fulltf+strat+leo+contre_econtr0.1t0.5_moeenc8e2k256h_shared256h_lb0.01_envpcc_8192gbs_0.0001lr_1e-05wd_3000epochs_200es_1gxe_4heads_256emb_0.15do__s1_j4968492_scored_predictions.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${DATA_ROOT}/results/parabra}
RUN_ID=${RUN_ID:-parabra-python-zsemf-j${SLURM_JOB_ID}}
OUTPUT_DIR=${OUTPUT_ROOT}/${RUN_ID}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

echo "Output directory: ${OUTPUT_DIR}"
srun -N 1 -n 1 -c "${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    "${PYTHON_BIN}" -u scripts/train_parabra.py \
    --data-root "${DATA_ROOT}" \
    --transformer-predictions "${TRANSFORMER_PREDICTIONS}" \
    --output-dir "${OUTPUT_DIR}"

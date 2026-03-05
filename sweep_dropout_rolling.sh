#!/bin/bash
# bash sweep_dropout_rolling.sh          # submit all 5 jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/train_rolling.slurm"
SWEEP_GROUP="rolling-vs-test-$(date +%Y%m%d-%H%M%S)"

DRY_RUN=false
if [[ "${1:-}" == "--dry" || "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Shared training budget (micro-runs — just need relative ranking) ──
SWEEP_NODES=4
SWEEP_TIME="02:00:00"
SWEEP_PARTITION="batch"
SWEEP_NUM_EPOCHS=100
SWEEP_EARLY_STOP=15
SWEEP_GBS=8192

# ── Config definitions ──
# Format: TAG|DROPOUT|EMB_SIZE|G_ENCODER_TYPE|FULL_TF_MLP_TYPE|MOE_NUM_EXPERTS|MOE_SHARED_EXPERT|DESCRIPTION
CONFIGS=(
    "baseline|0.15|256|moe|moe|8|True|current best config"
    "tiny|0.10|64|dense|dense|0|False|tiny dense model"
    "huge_drop|0.50|256|moe|moe|8|True|over-regularized"
    "no_moe|0.15|256|dense|dense|0|False|dense MLP no MoE"
    "zero_drop|0.00|256|moe|moe|8|True|under-regularized"
)

echo "=================================================================="
echo "Rolling CV vs 2024 Test — Quick Validation Sweep"
echo "=================================================================="
echo "Group        : ${SWEEP_GROUP}"
echo "Configs      : ${#CONFIGS[@]}"
echo "Folds        : val 2019-2023 (5 micro-folds)"
echo "Epochs       : ${SWEEP_NUM_EPOCHS} (early stop: ${SWEEP_EARLY_STOP})"
echo "Nodes/job    : ${SWEEP_NODES}  wall: ${SWEEP_TIME}"
echo "SLURM base   : ${SLURM_TEMPLATE}"
echo "Dry run      : ${DRY_RUN}"
echo "=================================================================="
echo ""

mkdir -p logs/sweeps

JOB_IDS=()
for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG DROPOUT EMB G_ENC_TYPE TF_MLP_TYPE N_EXPERTS SHARED_EXP DESC <<< "$config_str"
    echo "[SUBMIT] ${TAG}: ${DESC}"
    echo "         dropout=${DROPOUT} emb=${EMB} g_enc=${G_ENC_TYPE} mlp=${TF_MLP_TYPE} experts=${N_EXPERTS}"

    # When using dense, set expert count to a safe default (ignored by model)
    EXPERT_DIM=${EMB}
    SHARED_DIM=${EMB}

    # Export vars that contain commas BEFORE sbatch (--export uses comma as delimiter)
    export ROLLING_VAL_YEARS="2019,2020,2021,2022,2023"
    export ROLLING_SINGLE_VAL_YEAR=""

    SBATCH_CMD=(
        sbatch
        -N "${SWEEP_NODES}"
        -t "${SWEEP_TIME}"
        -p "${SWEEP_PARTITION}"
        --job-name="sweep-${TAG}"
        --output="logs/sweeps/${SWEEP_GROUP}-${TAG}-%j.out"
        --error="logs/sweeps/${SWEEP_GROUP}-${TAG}-%j.err"
        --export="ALL,\
DROPOUT=${DROPOUT},\
EMB_SIZE=${EMB},\
G_ENCODER_TYPE=${G_ENC_TYPE},\
FULL_TF_MLP_TYPE=${TF_MLP_TYPE},\
MOE_NUM_EXPERTS=${N_EXPERTS},\
MOE_SHARED_EXPERT=${SHARED_EXP},\
MOE_EXPERT_HIDDEN_DIM=${EXPERT_DIM},\
MOE_SHARED_EXPERT_HIDDEN_DIM=${SHARED_DIM},\
GBS=${SWEEP_GBS},\
NUM_EPOCHS=${SWEEP_NUM_EPOCHS},\
EARLY_STOP=${SWEEP_EARLY_STOP},\
ROLLING_FULL_CV=False,\
ROLLING_MAX_FOLDS=5,\
ROLLING_RECENT_FIRST=False,\
ROLLING_PARALLEL_OUTER_FOLDS=False,\
ROLLING_TEST_EVAL_MODE=all_folds,\
ROLLING_TEST_PRIMARY=best_fold,\
ROLLING_WANDB_NAME_SUFFIX=sweep+${TAG},\
WANDB_TAGS=${SWEEP_GROUP},\
SWEEP_GROUP=${SWEEP_GROUP},\
SWEEP_TAG=${TAG}"
        "${SLURM_TEMPLATE}"
    )

    if $DRY_RUN; then
        echo "  [DRY] ${SBATCH_CMD[*]}"
    else
        JOB_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1)
        JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+' | tail -1)
        JOB_IDS+=("$JOB_ID")
        echo "  -> Job ${JOB_ID}"
    fi
    echo ""
done

# Save sweep metadata
META_FILE="logs/sweeps/${SWEEP_GROUP}-meta.txt"
{
    echo "sweep_group=${SWEEP_GROUP}"
    echo "submitted=$(date -Iseconds)"
    echo "configs=${#CONFIGS[@]}"
    echo "num_epochs=${SWEEP_NUM_EPOCHS}"
    echo "early_stop=${SWEEP_EARLY_STOP}"
    echo "nodes_per_job=${SWEEP_NODES}"
    echo "val_years=2019,2020,2021,2022,2023"
    echo "wandb_project=gxe-transformer-rolling"
    echo "wandb_entity=$(grep WANDB_ENTITY .env 2>/dev/null | cut -d= -f2 | tr -d '"' || echo 'unknown')"
    for config_str in "${CONFIGS[@]}"; do
        IFS='|' read -r TAG REST <<< "$config_str"
        echo "config_${TAG}=${config_str}"
    done
    if [ ${#JOB_IDS[@]} -gt 0 ]; then
        echo "job_ids=${JOB_IDS[*]}"
    fi
} > "$META_FILE"

echo "=================================================================="
echo "Sweep submitted (${#CONFIGS[@]} jobs × ${SWEEP_NODES} nodes × ${SWEEP_TIME})."
echo "Metadata: ${META_FILE}"
echo ""
echo "After all jobs finish, run:"
echo "  python scripts/analyze_rolling_vs_test.py --group ${SWEEP_GROUP}"
echo "=================================================================="

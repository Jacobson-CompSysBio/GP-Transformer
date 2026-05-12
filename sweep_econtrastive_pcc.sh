#!/bin/bash
# bash sweep_econtrastive_pcc.sh --dry
# bash sweep_econtrastive_pcc.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/best_train.slurm"
SWEEP_GROUP="gxe-pcc-econtrastive-$(date +%Y%m%d-%H%M%S)"

DRY_RUN=false
if [[ "${1:-}" == "--dry" || "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# TAG|E_WEIGHT|E_TEMP|WARMUP|RAMP|CAL_MODE|VAL_SCHEME|PROXY_FRAC|ENVCCC|HUBER|DESC
CONFIGS=(
    "base|0.10|0.50|50|50|none|leo|0.25|0.0|0.0|current PCC-first trunk"
    "elow|0.05|0.50|50|50|none|leo|0.25|0.0|0.0|lower env contrastive weight"
    "ehigh|0.20|0.50|50|50|none|leo|0.25|0.0|0.0|higher env contrastive weight"
    "tcool|0.10|0.25|50|50|none|leo|0.25|0.0|0.0|sharper env contrastive temperature"
    "twarm|0.10|1.00|50|50|none|leo|0.25|0.0|0.0|softer env contrastive temperature"
    "slowramp|0.10|0.50|50|150|none|leo|0.25|0.0|0.0|slower contrastive ramp"
    "affcal|0.10|0.50|50|50|env_affine|leo|0.25|0.05|0.05|rank-safe scale diagnostic"
    "proxydiag|0.10|0.50|50|50|none|proxy_same_tester|0.25|0.0|0.0|same-tester proxy diagnostic"
)

echo "=================================================================="
echo "GxE PCC-first environment-contrastive sweep"
echo "=================================================================="
echo "Group      : ${SWEEP_GROUP}"
echo "Configs    : ${#CONFIGS[@]}"
echo "Template   : ${SLURM_TEMPLATE}"
echo "Dry run    : ${DRY_RUN}"
echo "Selection  : train.py best_leo alias; y_test is eval-only"
echo "=================================================================="
echo ""

mkdir -p logs/sweeps

JOB_IDS=()
for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG E_WEIGHT E_TEMP WARMUP RAMP CAL_MODE VAL_SCHEME PROXY_FRAC ENVCCC HUBER DESC <<< "$config_str"
    echo "[SUBMIT] ${TAG}: ${DESC}"
    echo "         e_weight=${E_WEIGHT} e_temp=${E_TEMP} warmup=${WARMUP} ramp=${RAMP} val=${VAL_SCHEME} cal=${CAL_MODE}"

    SBATCH_CMD=(
        sbatch
        --job-name="gxe-${TAG}"
        --output="logs/sweeps/${SWEEP_GROUP}-${TAG}-%j.out"
        --error="logs/sweeps/${SWEEP_GROUP}-${TAG}-%j.err"
        --export="ALL,\
CONTRASTIVE_MODE=e,\
ENV_CONTRASTIVE_WEIGHT=${E_WEIGHT},\
ENV_CONTRASTIVE_TEMPERATURE=${E_TEMP},\
CONTRASTIVE_WARMUP_EPOCHS=${WARMUP},\
CONTRASTIVE_RAMP_EPOCHS=${RAMP},\
VAL_SCHEME=${VAL_SCHEME},\
LEO_VAL=True,\
PROXY_TESTER=PHP02,\
PROXY_HOLDOUT_FRAC=${PROXY_FRAC},\
CALIBRATION_MODE=${CAL_MODE},\
ENVCCC_WEIGHT=${ENVCCC},\
HUBER_WEIGHT=${HUBER},\
CHECKPOINT_TAG=best_leo,\
RUN_NAME_SUFFIX=${SWEEP_GROUP}+${TAG},\
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

META_FILE="logs/sweeps/${SWEEP_GROUP}-meta.txt"
{
    echo "sweep_group=${SWEEP_GROUP}"
    echo "submitted=$(date -Iseconds)"
    echo "configs=${#CONFIGS[@]}"
    echo "template=${SLURM_TEMPLATE}"
    for config_str in "${CONFIGS[@]}"; do
        IFS='|' read -r TAG REST <<< "$config_str"
        echo "config_${TAG}=${config_str}"
    done
    if [ ${#JOB_IDS[@]} -gt 0 ]; then
        echo "job_ids=${JOB_IDS[*]}"
    fi
} > "$META_FILE"

echo "=================================================================="
echo "Sweep submitted. Metadata: ${META_FILE}"
echo "Primary comparison: data/results/*_test_metrics.csv and W&B env_avg_pearson."
echo "Promote only checkpoints that preserve or improve macro env PCC."
echo "=================================================================="

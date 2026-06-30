#!/bin/bash
# Rank-first SOTA campaign launcher.
#
# Examples:
#   bash sweep_econtrastive_pcc.sh --dry --phase baseline
#   bash sweep_econtrastive_pcc.sh --phase grid
#   TOP_CONFIGS="top1:0.1:0.5,top2:0.2:0.25,top3:0.15:1.0" \
#     bash sweep_econtrastive_pcc.sh --phase repeat-top
#   BEST_RANK_CONFIG="best:0.1:0.5" bash sweep_econtrastive_pcc.sh --phase schedule
#   RANK_CONFIGS="rank1:0.1:0.5:50:50,rank2:0.2:0.25:50:50" \
#     bash sweep_econtrastive_pcc.sh --phase calibration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_TEMPLATE="${SCRIPT_DIR}/best_train.slurm"
SWEEP_GROUP="${SWEEP_GROUP:-gxe-rankfirst-$(date +%Y%m%d-%H%M%S)}"
PHASE="grid"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --group)
            SWEEP_GROUP="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '1,80p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

CONFIGS=()

add_config() {
    # tag|seed|e_weight|e_temp|warmup|ramp|cal_mode|envccc|huber|proxy_mode|desc
    CONFIGS+=("$1|$2|$3|$4|$5|$6|$7|$8|$9|${10}|${11}")
}

split_rank_config() {
    local spec="$1"
    IFS=':' read -r TAG E_WEIGHT E_TEMP WARMUP RAMP EXTRA <<< "$spec"
    WARMUP="${WARMUP:-50}"
    RAMP="${RAMP:-50}"
    echo "${TAG}|${E_WEIGHT}|${E_TEMP}|${WARMUP}|${RAMP}|${EXTRA:-}"
}

case "$PHASE" in
    baseline)
        for SEED in 1 2 3; do
            add_config "baseline-s${SEED}" "$SEED" "0.10" "0.50" "50" "50" "none" "0.10" "0.02" "same_tester_novel_cross" "baseline repeat"
        done
        ;;
    grid)
        for E_WEIGHT in 0.05 0.10 0.15 0.20; do
            for E_TEMP in 0.25 0.50 1.00; do
                TAG="grid-w${E_WEIGHT//./p}-t${E_TEMP//./p}-s1"
                add_config "$TAG" "1" "$E_WEIGHT" "$E_TEMP" "50" "50" "none" "0.10" "0.02" "same_tester_novel_cross" "seed-1 weight/temp sweep"
            done
        done
        ;;
    repeat-top)
        if [[ -z "${TOP_CONFIGS:-}" ]]; then
            echo "[ERROR] repeat-top requires TOP_CONFIGS='tag:weight:temp[,tag:weight:temp...]'" >&2
            exit 2
        fi
        IFS=',' read -ra SPECS <<< "$TOP_CONFIGS"
        for SPEC in "${SPECS[@]}"; do
            IFS='|' read -r TAG E_WEIGHT E_TEMP WARMUP RAMP _ <<< "$(split_rank_config "$SPEC")"
            for SEED in 2 3; do
                add_config "${TAG}-repeat-s${SEED}" "$SEED" "$E_WEIGHT" "$E_TEMP" "$WARMUP" "$RAMP" "none" "0.10" "0.02" "same_tester_novel_cross" "top-grid repeat"
            done
        done
        ;;
    schedule)
        BEST_RANK_CONFIG="${BEST_RANK_CONFIG:-best:0.10:0.50}"
        IFS='|' read -r TAG E_WEIGHT E_TEMP _ _ _ <<< "$(split_rank_config "$BEST_RANK_CONFIG")"
        for SCHED in "0:50" "25:50" "50:50" "50:100"; do
            IFS=':' read -r WARMUP RAMP <<< "$SCHED"
            add_config "${TAG}-sched-wu${WARMUP}-ra${RAMP}-s1" "1" "$E_WEIGHT" "$E_TEMP" "$WARMUP" "$RAMP" "none" "0.10" "0.02" "same_tester_novel_cross" "schedule sweep"
        done
        ;;
    repeat-schedule)
        if [[ -z "${BEST_SCHEDULE_CONFIG:-}" ]]; then
            echo "[ERROR] repeat-schedule requires BEST_SCHEDULE_CONFIG='tag:weight:temp:warmup:ramp'" >&2
            exit 2
        fi
        IFS='|' read -r TAG E_WEIGHT E_TEMP WARMUP RAMP _ <<< "$(split_rank_config "$BEST_SCHEDULE_CONFIG")"
        for SEED in 2 3; do
            add_config "${TAG}-sched-repeat-s${SEED}" "$SEED" "$E_WEIGHT" "$E_TEMP" "$WARMUP" "$RAMP" "none" "0.10" "0.02" "same_tester_novel_cross" "best schedule repeat"
        done
        ;;
    calibration)
        RANK_CONFIGS="${RANK_CONFIGS:-rank1:0.10:0.50:50:50,rank2:0.15:0.50:50:50}"
        IFS=',' read -ra SPECS <<< "$RANK_CONFIGS"
        for SPEC in "${SPECS[@]}"; do
            IFS='|' read -r TAG E_WEIGHT E_TEMP WARMUP RAMP _ <<< "$(split_rank_config "$SPEC")"
            for HUBER in 0.01 0.02 0.05; do
                HTAG="${HUBER//./p}"
                add_config "${TAG}-affcal-h${HTAG}-s1" "1" "$E_WEIGHT" "$E_TEMP" "$WARMUP" "$RAMP" "env_affine" "0.10" "$HUBER" "same_tester_novel_cross" "env-affine calibration sweep"
            done
        done
        ;;
    repeat-calibration)
        if [[ -z "${BEST_CAL_CONFIG:-}" ]]; then
            echo "[ERROR] repeat-calibration requires BEST_CAL_CONFIG='tag:weight:temp:warmup:ramp:huber'" >&2
            exit 2
        fi
        IFS='|' read -r TAG E_WEIGHT E_TEMP WARMUP RAMP HUBER <<< "$(split_rank_config "$BEST_CAL_CONFIG")"
        HUBER="${HUBER:-0.02}"
        for SEED in 2 3; do
            add_config "${TAG}-affcal-repeat-s${SEED}" "$SEED" "$E_WEIGHT" "$E_TEMP" "$WARMUP" "$RAMP" "env_affine" "0.10" "$HUBER" "same_tester_novel_cross" "best calibration repeat"
        done
        ;;
    all)
        "$0" ${DRY_RUN:+--dry} --group "${SWEEP_GROUP}-baseline" --phase baseline
        "$0" ${DRY_RUN:+--dry} --group "${SWEEP_GROUP}-grid" --phase grid
        echo "[INFO] Use repeat-top/schedule/calibration after ranking completed runs."
        exit 0
        ;;
    *)
        echo "[ERROR] Unknown phase: $PHASE" >&2
        exit 2
        ;;
esac

echo "=================================================================="
echo "GxE rank-first SOTA campaign"
echo "=================================================================="
echo "Group     : ${SWEEP_GROUP}"
echo "Phase     : ${PHASE}"
echo "Configs   : ${#CONFIGS[@]}"
echo "Template  : ${SLURM_TEMPLATE}"
echo "Dry run   : ${DRY_RUN}"
echo "Selector  : best_leo / val_loss/env_avg_pearson"
echo "Proxy     : logged as val_proxy/* only, not a checkpoint selector"
echo "=================================================================="
echo ""

mkdir -p logs/sweeps
JOB_IDS=()

for config_str in "${CONFIGS[@]}"; do
    IFS='|' read -r TAG SEED E_WEIGHT E_TEMP WARMUP RAMP CAL_MODE ENVCCC HUBER PROXY_MODE DESC <<< "$config_str"
    echo "[SUBMIT] ${TAG}: ${DESC}"
    echo "         seed=${SEED} e_weight=${E_WEIGHT} e_temp=${E_TEMP} warmup=${WARMUP} ramp=${RAMP} cal=${CAL_MODE} huber=${HUBER}"

    SBATCH_CMD=(
        sbatch
        --job-name="gxe-${TAG}"
        --output="logs/sweeps/${SWEEP_GROUP}-${TAG}-%j.out"
        --error="logs/sweeps/${SWEEP_GROUP}-${TAG}-%j.err"
        --export="ALL,\
SEED=${SEED},\
CONTRASTIVE_MODE=e,\
ENV_CONTRASTIVE_WEIGHT=${E_WEIGHT},\
ENV_CONTRASTIVE_TEMPERATURE=${E_TEMP},\
CONTRASTIVE_WARMUP_EPOCHS=${WARMUP},\
CONTRASTIVE_RAMP_EPOCHS=${RAMP},\
VAL_SCHEME=leo,\
LEO_VAL=True,\
PROXY_VALIDATION_MODE=${PROXY_MODE},\
PROXY_TESTER=PHP02,\
PROXY_HOLDOUT_FRAC=0.20,\
PROXY_DISJOINT_FROM_LEO=True,\
CALIBRATION_MODE=${CAL_MODE},\
CALIBRATION_START_EPOCH=150,\
CALIBRATION_RAMP_EPOCHS=100,\
CALIBRATION_DETACH_RANK_UNTIL_EPOCH=250,\
CALIBRATION_JOINT_GRAD_FRACTION=0.10,\
ENVCCC_WEIGHT=${ENVCCC},\
HUBER_WEIGHT=${HUBER},\
CHECKPOINT_TAG=best_leo,\
RUN_NAME_SUFFIX=${SWEEP_GROUP}+${TAG},\
WANDB_TAGS=${SWEEP_GROUP},\
SWEEP_GROUP=${SWEEP_GROUP},\
SWEEP_PHASE=${PHASE},\
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

META_FILE="logs/sweeps/${SWEEP_GROUP}-${PHASE}-meta.txt"
{
    echo "sweep_group=${SWEEP_GROUP}"
    echo "phase=${PHASE}"
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
echo "Sweep metadata: ${META_FILE}"
echo "Rank promotion: 3-seed mean env-PCC >= baseline + 0.003, or single-run > 0.43 with repeat support."
echo "Scale promotion: env-MSE improves while env-PCC loss <= 0.002 against matched rank model."
echo "=================================================================="

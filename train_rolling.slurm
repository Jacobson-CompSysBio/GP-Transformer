#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J train-rolling
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -eo pipefail
ulimit -S -c 0

module purge
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

# Rolling CV guide:
#   - See ROLLING_TRAINING.md for variable semantics + launch recipes.

### model / training hyperparameters (same surface as train.py/train_residual.py) ###
export G_ENC=True
export E_ENC=True
export LD_ENC=True
export GXE_ENC=tf
export WG=True
export FULL_TRANSFORMER=True
export RES=False

export G_ENCODER_TYPE=moe
export FULL_TF_MLP_TYPE=moe
export MOE_NUM_EXPERTS=8
export MOE_TOP_K=2
export MOE_SHARED_EXPERT=True
export MOE_EXPERT_HIDDEN_DIM=256
export MOE_SHARED_EXPERT_HIDDEN_DIM=256
export MOE_LOSS_WEIGHT=0.01

export GBS=8192
NGPUS=$(($SLURM_NNODES * 8))
MBS=$(($GBS / $NGPUS))
export BATCH_SIZE=$MBS
echo "Rolling CV with global batch size $GBS on $NGPUS GPUs (microbatch size $MBS)"

export NUM_EPOCHS=1200
export LR=1e-4
export WEIGHT_DECAY=1e-5
export DROPOUT=0.15
export EARLY_STOP=100

export G_LAYERS=1
export LD_LAYERS=1
export MLP_LAYERS=1
export GXE_LAYERS=1

export HEADS=4
export EMB_SIZE=256
export SEED=1
export SCALE_TARGETS=True
export G_INPUT_TYPE=tokens

export LOSS="envpcc"
export LOSS_WEIGHTS="1.0"

export DETACH_YMEAN=True
export LAMBDA_YMEAN=0.5
export LAMBDA_RESID=1.0

export ENV_STRATIFIED=True
export MIN_SAMPLES_PER_ENV=32

export CONTRASTIVE_MODE=none
export CONTRASTIVE_WEIGHT=0.1
export CONTRASTIVE_TEMPERATURE=0.1
export CONTRASTIVE_SIM_TYPE=grm
export CONTRASTIVE_LOSS_TYPE=mse
export ENV_CONTRASTIVE_WEIGHT=0.1
export ENV_CONTRASTIVE_TEMPERATURE=0.5

### rolling controls ###
# Current default below is a tractable recent-years subset (2021-2023).
# For full CV, set:
#   ROLLING_FULL_CV=True
#   ROLLING_VAL_YEARS=""
#   ROLLING_MAX_FOLDS=""
# Fold construction (applies when ROLLING_PARALLEL_OUTER_FOLDS=False):
#   - ROLLING_FULL_CV=True:
#       Use all forward folds in [ROLLING_TRAIN_START_YEAR, ROLLING_TRAIN_END_YEAR),
#       e.g. 2014->2023 means val years 2015..2023.
#   - ROLLING_FULL_CV=False + ROLLING_VAL_YEARS="2021,2022,2023":
#       Use only those val years.
#   - ROLLING_RECENT_FIRST=True:
#       Reverse fold order before truncation.
#   - ROLLING_MAX_FOLDS=K:
#       Keep first K folds after ordering.
#   - ROLLING_SINGLE_VAL_YEAR=YYYY:
#       Run exactly one fold (overrides ROLLING_MAX_FOLDS).
export ROLLING_FULL_CV=True
export ROLLING_VAL_YEARS=""
export ROLLING_TRAIN_START_YEAR=2014
export ROLLING_TRAIN_END_YEAR=2023
export ROLLING_MAX_FOLDS=3
export ROLLING_RECENT_FIRST=False
export ROLLING_SINGLE_VAL_YEAR=""
export ROLLING_WANDB_NAME_SUFFIX=""

# Outer-fold parallel mode (submit as SLURM array):
#   - Each array task trains exactly one fold.
#   - ROLLING_PARALLEL_VAL_YEARS controls mapped val years. If empty,
#     fallback is ROLLING_VAL_YEARS, else full [start+1 .. end] range.
#   - ROLLING_PARALLEL_MAX_CONCURRENT is used in the printed resubmit hint.
export ROLLING_PARALLEL_OUTER_FOLDS=False
export ROLLING_PARALLEL_MAX_CONCURRENT=3
export ROLLING_PARALLEL_VAL_YEARS=""

# rolling test-eval controls
# options: none | best_fold | latest_fold | best_and_latest | all_folds | year:2023
# recommendation:
#   - during model selection CV runs: use ROLLING_TEST_EVAL_MODE=none
#   - final reporting run: set to best_fold (or all_folds for diagnostics)
export ROLLING_TEST_EVAL_MODE=best_fold
export ROLLING_TEST_BATCH_SIZE=32
export ROLLING_TEST_PRIMARY=best_fold

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

if is_true "${ROLLING_PARALLEL_OUTER_FOLDS:-False}"; then
    declare -a VAL_YEARS=()
    years_src="${ROLLING_PARALLEL_VAL_YEARS:-}"
    if [ -z "${years_src//[[:space:]]/}" ]; then
        years_src="${ROLLING_VAL_YEARS:-}"
    fi

    if [ -n "${years_src//[[:space:]]/}" ]; then
        IFS=',' read -r -a _raw_years <<< "$years_src"
        for _year in "${_raw_years[@]}"; do
            _year="${_year//[[:space:]]/}"
            [ -n "$_year" ] && VAL_YEARS+=("$_year")
        done
    else
        for ((y=ROLLING_TRAIN_START_YEAR + 1; y<=ROLLING_TRAIN_END_YEAR; y++)); do
            VAL_YEARS+=("$y")
        done
    fi

    if [ "${#VAL_YEARS[@]}" -eq 0 ]; then
        echo "[ERROR] ROLLING_PARALLEL_OUTER_FOLDS=True but no val years were resolved."
        exit 2
    fi

    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        max_idx=$((${#VAL_YEARS[@]} - 1))
        echo "[ERROR] Parallel outer-fold mode requires SLURM array submission."
        echo "[INFO] Resubmit using:"
        echo "       sbatch --array=0-${max_idx}%${ROLLING_PARALLEL_MAX_CONCURRENT} $0"
        echo "[INFO] Resolved val years: ${VAL_YEARS[*]}"
        exit 2
    fi

    fold_idx=${SLURM_ARRAY_TASK_ID}
    if [ "$fold_idx" -lt 0 ] || [ "$fold_idx" -ge "${#VAL_YEARS[@]}" ]; then
        echo "[ERROR] SLURM_ARRAY_TASK_ID=$fold_idx out of range for ${#VAL_YEARS[@]} folds."
        exit 2
    fi

    export ROLLING_SINGLE_VAL_YEAR="${VAL_YEARS[$fold_idx]}"
    export ROLLING_FULL_CV=False
    export ROLLING_VAL_YEARS="${ROLLING_SINGLE_VAL_YEAR}"
    export ROLLING_MAX_FOLDS=1
    export ROLLING_RECENT_FIRST=False
    export ROLLING_WANDB_NAME_SUFFIX="outer${ROLLING_SINGLE_VAL_YEAR}"

    echo "[INFO] Parallel outer-fold mode: task ${fold_idx} -> val year ${ROLLING_SINGLE_VAL_YEAR}"
fi

### wandb settings ###
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

mkdir -p logs/ckpt_ids logs/run_ids checkpoints data/results
RUN_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-na}"
export WANDB_RUN_ID_FILE="logs/run_ids/wandb_run_id_${RUN_TAG}.txt"
export CHECKPOINT_DIR_FILE="logs/ckpt_ids/checkpoint_dir_${RUN_TAG}.txt"
export ROLLING_SELECTED_CHECKPOINT_FILE="logs/ckpt_ids/selected_checkpoint_${RUN_TAG}.txt"
rm -f "$WANDB_RUN_ID_FILE" "$CHECKPOINT_DIR_FILE"
rm -f "$ROLLING_SELECTED_CHECKPOINT_FILE"

### network / rccl settings ###
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=hsn0

### miopen settings ###
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH" && mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=NORMAL

### srun ###
srun -N ${SLURM_NNODES} \
    -n ${NGPUS} \
    --gpus-per-task=1 \
    --cpu-bind=cores \
    --export=ALL \
    --kill-on-bad-exit=1 \
    python -u scripts/train_rolling.py \
        --batch_size $BATCH_SIZE \
        --gbs $GBS \
        --g_enc $G_ENC \
        --e_enc $E_ENC \
        --ld_enc $LD_ENC \
        --gxe_enc $GXE_ENC \
        --wg $WG \
        --full_transformer $FULL_TRANSFORMER \
        --residual $RES \
        --g_encoder_type $G_ENCODER_TYPE \
        --full_tf_mlp_type $FULL_TF_MLP_TYPE \
        --moe_num_experts $MOE_NUM_EXPERTS \
        --moe_top_k $MOE_TOP_K \
        --moe_shared_expert $MOE_SHARED_EXPERT \
        --moe_expert_hidden_dim $MOE_EXPERT_HIDDEN_DIM \
        --moe_shared_expert_hidden_dim $MOE_SHARED_EXPERT_HIDDEN_DIM \
        --moe_loss_weight $MOE_LOSS_WEIGHT \
        --detach_ymean $DETACH_YMEAN \
        --lambda_ymean $LAMBDA_YMEAN \
        --lambda_resid $LAMBDA_RESID \
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
        --seed $SEED \
        --loss $LOSS \
        --loss_weights $LOSS_WEIGHTS \
        --scale_targets $SCALE_TARGETS \
        --g_input_type $G_INPUT_TYPE \
        --env_stratified $ENV_STRATIFIED \
        --min_samples_per_env $MIN_SAMPLES_PER_ENV \
        --contrastive_mode $CONTRASTIVE_MODE \
        --contrastive_weight $CONTRASTIVE_WEIGHT \
        --contrastive_temperature $CONTRASTIVE_TEMPERATURE \
        --contrastive_sim_type $CONTRASTIVE_SIM_TYPE \
        --contrastive_loss_type $CONTRASTIVE_LOSS_TYPE \
        --env_contrastive_weight $ENV_CONTRASTIVE_WEIGHT \
        --env_contrastive_temperature $ENV_CONTRASTIVE_TEMPERATURE

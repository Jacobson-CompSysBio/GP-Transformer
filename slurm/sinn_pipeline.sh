#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SINN Pipeline: Submit all 4 phases as dependent SLURM jobs
#
# Usage:
#   ./slurm/sinn_pipeline.sh              # auto-generate tag
#   ./slurm/sinn_pipeline.sh my-run-01    # custom tag
#
# Each phase depends on the previous one completing successfully.
# Checkpoint paths are chained via SINN_TAG files in logs/ckpt_ids/.
# You can also submit phases individually:
#   SINN_TAG=my-run-01 sbatch slurm/sinn_g.slurm
#   SINN_TAG=my-run-01 sbatch slurm/sinn_finetune.slurm   # after GE done
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SINN_TAG="${1:-sinn-$(date +%Y%m%d-%H%M%S)}"
echo "═══════════════════════════════════════════"
echo " SINN Pipeline — Tag: ${SINN_TAG}"
echo "═══════════════════════════════════════════"

# Phase G (1 node, ~1 hr)
JOB_G=$(sbatch --parsable --export=ALL,SINN_TAG=${SINN_TAG} slurm/sinn_g.slurm)
echo "[G]        Job ${JOB_G} submitted (1 node)"

# Phase E (1 node, ~2 hrs) — independent of G, can run in parallel
JOB_E=$(sbatch --parsable --export=ALL,SINN_TAG=${SINN_TAG} slurm/sinn_e.slurm)
echo "[E]        Job ${JOB_E} submitted (1 node)"

# Phase GE (1 node, ~2 hrs) — depends on G and E
JOB_GE=$(sbatch --parsable --export=ALL,SINN_TAG=${SINN_TAG} \
    --dependency=afterok:${JOB_G}:${JOB_E} slurm/sinn_ge.slurm)
echo "[GE]       Job ${JOB_GE} submitted (1 node, after G+E)"

# Phase Finetune (2 nodes, ~4 hrs) — depends on GE
JOB_FT=$(sbatch --parsable --export=ALL,SINN_TAG=${SINN_TAG} \
    --dependency=afterok:${JOB_GE} slurm/sinn_finetune.slurm)
echo "[Finetune] Job ${JOB_FT} submitted (4 nodes, after GE)"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/sinn-g-${JOB_G}.out"
echo "  tail -f logs/sinn-ft-${JOB_FT}.out"
echo ""
echo "Checkpoints will be in: checkpoints/sinn-${SINN_TAG}/"

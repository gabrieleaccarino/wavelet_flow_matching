#!/bin/bash
# ============================================================================
# Submit the full IC-perturbation sweep: one PBS job per (SEED, IC_NOISE).
# Each job writes its own CSV (filename + seed/ic_noise columns identify it);
# concatenate them afterwards with aggregate_ic_results.py.
#
#   bash submit_ic_grid.sh
# ============================================================================
set -euo pipefail

DATASET_NAME="turbulent_radiative_layer_2D"
REPO=/glade/work/gaccarino/waveletflux
PBS_SCRIPT=${REPO}/scripts/inference_ic_pert.pbs
LOGDIR=${REPO}/scripts/logs/${DATASET_NAME}
mkdir -p "${LOGDIR}"

# ── the sweep grid ──────────────────────────────────────────────────────────
SEEDS=(42 84 168)                       # three seeds -> error bars
NOISES=(0.0 0.001 0.01 0.1)             # ic_noise levels (0.0 = baseline)

echo "Submitting $(( ${#SEEDS[@]} * ${#NOISES[@]} )) jobs "
echo "  seeds : ${SEEDS[*]}"
echo "  noise : ${NOISES[*]}"
echo

#WaveletFlow_turbulent_radiative_layer_2D_3scales_haar_v4

for SEED in "${SEEDS[@]}"; do
  for IC in "${NOISES[@]}"; do
    IC_TAG="ic${IC//./p}"                       # 0.05 -> ic0p05
    NAME="WaveletFlow_turbulent_radiative_layer_2D_3scales_haar_v4.0_${IC_TAG}_seed${SEED}"
    qsub \
      -N "${NAME}" \
      -o "${LOGDIR}/${NAME}.out" \
      -e "${LOGDIR}/${NAME}.err" \
      -v "SEED=${SEED},IC_NOISE=${IC}" \
      "${PBS_SCRIPT}"
    echo "  submitted ${NAME}"
  done
done

echo
echo "Done. Watch with:  qstat -u \$USER"
echo "When finished, aggregate with:"
echo "  python ${REPO}/scripts/aggregate_ic_results.py \\"
echo "    --results_dir ${REPO}/experiments/${DATASET_NAME}/results"

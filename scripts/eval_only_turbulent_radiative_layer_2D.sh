#!/bin/bash -l
#PBS -N FM_data_space_eval_only_turbulent_radiative_layer_2D
#PBS -o /glade/work/gaccarino/wavelet_flow_matching/scripts/logs/turbulent_radiative_layer_2D/FM_data_space_eval_only_turbulent_radiative_layer_2D.out
#PBS -e /glade/work/gaccarino/wavelet_flow_matching/scripts/logs/turbulent_radiative_layer_2D/FM_data_space_eval_only_turbulent_radiative_layer_2D.err

# === Project / Queue ===
#PBS -A UCLB0062
#PBS -q main

# === Resources — CPU only, no GPUs needed ===
#PBS -l select=1:ncpus=16:mem=128GB
#PBS -l walltime=01:00:00

# === Move to submission directory ===
cd $PBS_O_WORKDIR

DATASET_NAME="turbulent_radiative_layer_2D"
REPO=/glade/work/gaccarino/wavelet_flow_matching
CONFIG=${REPO}/configs/${DATASET_NAME}/inference_${DATASET_NAME}.yaml
N_ENSEMBLE=8
NUM_SAMPLING=20

conda activate mlenv

echo "========================================================"
echo "  Job ID       : ${PBS_JOBID}"
echo "  Job name     : ${PBS_JOBNAME}"
echo "  Config       : ${CONFIG}"
echo "  N_ENSEMBLE   : ${N_ENSEMBLE}"
echo "  NUM_SAMPLING : ${NUM_SAMPLING}"
echo "  Started      : $(date)"
echo "========================================================"

mkdir -p ${REPO}/scripts/logs/${DATASET_NAME}

python ${REPO}/wavelet_flow_matching/eval_only.py \
    --config ${CONFIG} \
    --override \
        n_ensemble=${N_ENSEMBLE} \
        num_sampling_steps=${NUM_SAMPLING}

EXIT_CODE=$?

echo "========================================================"
echo "  Finished : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "========================================================"
exit ${EXIT_CODE}

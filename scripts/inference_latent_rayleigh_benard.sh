#!/bin/bash -l
#PBS -N LatentFM_rayleigh_benard_dcae_f32c64_v4.0_inference_seed168
#PBS -o /glade/work/gaccarino/waveletflux/scripts/logs/rayleigh_benard/LatentFM_rayleigh_benard_dcae_f32c64_v4.0_inference_seed168.out
#PBS -e /glade/work/gaccarino/waveletflux/scripts/logs/rayleigh_benard/LatentFM_rayleigh_benard_dcae_f32c64_v4.0_inference_seed168.err

# === Project / Queue ===
#PBS -A UCLB0062
#PBS -q main

# === Resources ===
#PBS -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB
#PBS -l walltime=01:00:00

# === Move to submission directory ===
cd $PBS_O_WORKDIR

DATASET_NAME="rayleigh_benard"
REPO=/glade/work/gaccarino/waveletflux
CONFIG=${REPO}/configs/${DATASET_NAME}/inference_${DATASET_NAME}_seed.yaml

# The inference driver is model-agnostic (dispatch via `model_class` in the
# YAML through waveletflux.utils.registry). This must be the registry-based
# inference file you edited.
INFER=${REPO}/waveletflux/inference_latent_fm.py

N_GPUS=4
N_ENSEMBLE=8
NUM_SAMPLING=50
ODE_METHOD="euler"        # keep in sync with the YAML's ode_method
CLAMP_SIGMA=5.0

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

conda activate mlenv

mkdir -p ${REPO}/scripts/logs/${DATASET_NAME}

echo "========================================================"
echo "  Job ID      : ${PBS_JOBID}"
echo "  Job name    : ${PBS_JOBNAME}"
echo "  Node list   : $(cat ${PBS_NODEFILE} | tr '\n' ' ')"
echo "  GPUs        : ${N_GPUS}"
echo "  Driver      : ${INFER}"
echo "  Config      : ${CONFIG}"
echo "  N_ENSEMBLE  : ${N_ENSEMBLE}"
echo "  NUM_SAMPLING: ${NUM_SAMPLING}"
echo "  ODE_METHOD  : ${ODE_METHOD}"
echo "  CLAMP_SIGMA : ${CLAMP_SIGMA}"
echo "  Started     : $(date)"
echo "========================================================"

torchrun \
    --nproc_per_node=${N_GPUS} \
    ${INFER} \
    --config ${CONFIG} \
    --override \
    n_ensemble=${N_ENSEMBLE} \
    num_sampling_steps=${NUM_SAMPLING} \
    ode_method=${ODE_METHOD} \
    clamp_sigma=${CLAMP_SIGMA}

EXIT_CODE=$?

echo "========================================================"
echo "  Finished : $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "========================================================"
exit ${EXIT_CODE}

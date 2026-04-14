#!/bin/bash -l
#PBS -N inference_planetswe
#PBS -o /glade/work/gaccarino/waveletflux/scripts/logs/inference_planetswe.out
#PBS -e /glade/work/gaccarino/waveletflux/scripts/logs/inference_planetswe.err

# === Project / Queue ===
#PBS -A UCLB0062
#PBS -q main

# === Resources ===
#PBS -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB
#PBS -l walltime=02:00:00

# === Move to submission directory ===
cd $PBS_O_WORKDIR

DATASET_NAME="planetswe"
REPO=/glade/work/gaccarino/waveletflux
CONFIG=${REPO}/configs/${DATASET_NAME}/inference_${DATASET_NAME}.yaml
N_GPUS=4
N_ENSEMBLE=8
NUM_SAMPLING=20

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

conda activate mlenv

echo "========================================================"
echo "  Job ID      : ${PBS_JOBID}"
echo "  Job name    : ${PBS_JOBNAME}"
echo "  Node list   : $(cat ${PBS_NODEFILE} | tr '\n' ' ')"
echo "  GPUs        : ${N_GPUS}"
echo "  Config      : ${CONFIG}"
echo "  N_ENSEMBLE  : ${N_ENSEMBLE}"
echo "  NUM_SAMPLING: ${NUM_SAMPLING}"
echo "  Started     : $(date)"
echo "========================================================"

mkdir -p ${REPO}/scripts/logs

torchrun \
    --nproc_per_node=${N_GPUS} \
    ${REPO}/waveletflux/inference.py \
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

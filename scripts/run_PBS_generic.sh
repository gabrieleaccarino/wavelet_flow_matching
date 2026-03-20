#!/bin/bash -l
# run_PBS_generic.sh — parameterised PBS training template
#
# Usage (set env vars before qsub):
#   ENTRY=waveletflux/train.py \
#   CONFIG=configs/turbulent_radiative_layer_2D/WaveletFlow_turbulent_radiative_layer_2D_v0.0.toml \
#   NGPUS=4 NODES=1 \
#   qsub -N MyJobName \
#        -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB \
#        -l walltime=12:00:00 \
#        scripts/run_PBS_generic.sh
#
# Required:  CONFIG — path to TOML config relative to REPO root
# Optional:  ENTRY  — entry point script (default: waveletflux/train.py)
#            NGPUS  — GPUs per node       (default: 4)
#            NODES  — number of nodes     (default: 1)

# === Project / Queue ===
#PBS -A UCLB0062
#PBS -q main

# === Move to submission directory ===
cd $PBS_O_WORKDIR

# === NCCL Configuration ===
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# === DeepSpeed ===
export DS_SKIP_AUTOTUNE=1

# === Environment ===
conda activate mlenv

# === Variables with defaults ===
REPO=/glade/work/gaccarino/waveletflux
: "${ENTRY:=waveletflux/train.py}"
: "${CONFIG:?ERROR: CONFIG must be set (e.g. CONFIG=configs/turbulent_radiative_layer_2D/WaveletFlow_v0.0.toml)}"
: "${NGPUS:=4}"
: "${NODES:=1}"

# === Run ===
torchrun --nproc_per_node=${NGPUS} \
    ${REPO}/${ENTRY} \
    -c ${REPO}/${CONFIG} \
    -n ${NODES} -d ${NGPUS}

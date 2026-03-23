#!/bin/bash -l
#PBS -N WaveletFlow_turbulent_radiative_layer_2D_1scales_db2_v1.0
#PBS -o /glade/work/gaccarino/waveletflux/scripts/logs/WaveletFlow_turbulent_radiative_layer_2D_1scales_db2_v1.0.out
#PBS -e /glade/work/gaccarino/waveletflux/scripts/logs/WaveletFlow_turbulent_radiative_layer_2D_1scales_db2_v1.0.err

# === Project / Queue ===
#PBS -A UCLB0062
#PBS -q main

# === Resources ===
#PBS -l select=1:ncpus=32:mpiprocs=2:ngpus=2:gpu_type=a100:mem=400GB
#PBS -l walltime=12:00:00

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

# === Run ===
REPO=/glade/work/gaccarino/waveletflux
torchrun --nproc_per_node=2 \
    ${REPO}/waveletflux/train.py \
    -c ${REPO}/configs/turbulent_radiative_layer_2D/WaveletFlow_turbulent_radiative_layer_2D_1scales_db2_v1.0.toml \
    -n 1 -d 2
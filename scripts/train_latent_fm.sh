#!/bin/bash -l
#PBS -N LatentFM_rayleigh_benard_dcae_f32c64_v4.0
#PBS -o /glade/work/gaccarino/wavelet_flow_matching/scripts/logs/rayleigh_benard/LatentFM_rayleigh_benard_dcae_f32c64_v4.0.out
#PBS -e /glade/work/gaccarino/wavelet_flow_matching/scripts/logs/rayleigh_benard/LatentFM_rayleigh_benard_dcae_f32c64_v4.0.err

# === Project / Queue ===
#PBS -A UCLB0062
#PBS -q main

# === Resources ===
#PBS -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB
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
REPO=/glade/work/gaccarino/wavelet_flow_matching
torchrun --nproc_per_node=4 \
    ${REPO}/wavelet_flow_matching/train_latent_fm.py \
    -c ${REPO}/configs/rayleigh_benard/LatentFM_rayleigh_benard_dcae_f32c64_v4.0.toml \
    -n 1 -d 4

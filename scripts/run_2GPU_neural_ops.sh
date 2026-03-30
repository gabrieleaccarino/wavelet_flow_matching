#!/bin/bash -l
#PBS -N FNO_turbulent_radiative_layer_2D_v1.0
#PBS -o /glade/work/gaccarino/waveletflux/scripts/logs/FNO_turbulent_radiative_layer_2D_v1.0.out
#PBS -e /glade/work/gaccarino/waveletflux/scripts/logs/FNO_turbulent_radiative_layer_2D_v1.0.err

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
torchrun --nproc_per_node=2 ${REPO}/waveletflux/train_neural_ops.py \
    --model fno --modes 16 --n_blocks 4 --hidden 128 --epochs 500 --batch_size 16 --lr 5e-3 \
    --num_workers 16 --save_every 50 --n_steps_input 4 --n_steps_output 1 \
    --well_base_path /glade/work/gaccarino/data/datasets \
    --well_dataset turbulent_radiative_layer_2D \
    --stats_path /glade/work/gaccarino/data/datasets/turbulent_radiative_layer_2D/stats.yaml \
    --checkpoint_dir ${REPO}/experiments/turbulent_radiative_layer_2D/FNO_turbulent_radiative_layer_2D_v1.0 \
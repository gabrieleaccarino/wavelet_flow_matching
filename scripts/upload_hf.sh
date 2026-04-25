#!/bin/bash -l
#PBS -N upload_waveletflux_hf
#PBS -o /glade/work/gaccarino/waveletflux/scripts/logs/upload/upload_on_hf.out
#PBS -e /glade/work/gaccarino/waveletflux/scripts/logs/upload/upload_on_hf.err

# === Project / Queue ===
#PBS -A UCLB0062
#PBS -q main

# === Resources (CPU only — no GPUs) ===
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=01:00:00

# === Move to submission directory ===
cd $PBS_O_WORKDIR

# === Hugging Face ===
export HF_TOKEN="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"     
export HF_XET_HIGH_PERFORMANCE=1   # max upload throughput

# === Environment ===
conda activate mlenv

# === Run ===
python /glade/work/gaccarino/waveletflux/waveletflux/upload_on_hf.py

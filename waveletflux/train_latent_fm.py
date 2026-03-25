"""
train_latent.py — Training script for LatentFlowModule.

Diff from train.py
──────────────────
Three lines change relative to the wavelet_fm train.py:

  1. Import LatentFlowModule instead of WaveletFlowModule
  2. Parse vae_cfg  from config.vae  instead of wavelet_cfg from config.wavelet
  3. Pass  vae_cfg  to the module constructor

Everything else (datamodule, trainer, callbacks, loggers, experiment manager,
normalizer, field discovery) is IDENTICAL — copy-paste from your existing
train.py and apply the three changes below.

The three-line diff is marked with  # ← CHANGED  comments.
"""

import os
import sys
import datetime
import argparse

import fsspec
import munch
import toml
import torch
import yaml
import pytorch_lightning as pl
from the_well.data import WellDataModule
from the_well.data.normalization import ZScoreNormalization
from einops import rearrange

# ── project imports ───────────────────────────────────────────────────────────
# (adjust sys.path / package name to match your project layout)
from waveletflux.utils.configs import (
    VelocityNetConfig,
    FlowMatchingConfig,
    OptimizerConfig,
)
from waveletflux.utils.latent_configs import LatentVAEConfig                  # ← CHANGED (1a)
from waveletflux.modules.latent_fm import LatentFlowModule                            # ← CHANGED (1b)
from waveletflux.utils.well_data import WellDataModuleLightning
from waveletflux.utils.experiment import ExperimentManager


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser  (identical to train.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train LatentFlowModule")
    p.add_argument('--config', type=str, required=True,
                   help='Path to latent_fm_*.toml config file')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = munch.munchify(toml.load(args.config))

    pl.seed_everything(config.random.seed)
    torch.set_float32_matmul_precision(
        config.lightning.trainer.get('matmul_precision', 'high')
    )

    # ── Typed configs ─────────────────────────────────────────────────────────
    vae_cfg           = LatentVAEConfig.from_config(config.vae)               # ← CHANGED (2)
    velocity_net_cfg  = VelocityNetConfig.from_config(config.velocity_net)
    flow_matching_cfg = FlowMatchingConfig.from_config(
        config.flow_matching, config.sampling
    )
    optimizer_cfg     = OptimizerConfig.from_config(
        config.optimizer, config.scheduler
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_cfg = config.train

    raw_dm = WellDataModule(
        well_base_path    = config.path.well_base_path,
        well_dataset_name = config.path.well_dataset_name,
        batch_size        = train_cfg.batch_size,
        use_normalization = False,
        normalization_type= None,
        data_workers      = 0,
        world_size        = 1,
        rank              = 0,
    )
    raw_dm.setup('fit')

    # Field names and data shape
    field_names = []
    for order in sorted(raw_dm.train_dataset.field_names):
        field_names.extend(raw_dm.train_dataset.field_names[order])
    in_channels = len(field_names)

    sample_batch = next(iter(raw_dm.train_dataloader()))
    _, _, H, W, _ = sample_batch['input_fields'].shape
    data_shape = (H, W)

    # Normalizer
    with fsspec.open(config.path.stats_path, mode='r') as f:
        stats = yaml.safe_load(f)
    normalizer = ZScoreNormalization(
        stats,
        core_field_names          = raw_dm.train_dataset.core_field_names,
        core_constant_field_names = raw_dm.train_dataset.core_constant_field_names,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentFlowModule(                                                  # ← CHANGED (3)
        in_channels       = in_channels,
        field_names       = field_names,
        data_shape        = data_shape,
        n_steps_output    = train_cfg.n_steps_output,
        scale_weights     = config.loss.scale_weights,
        vae_cfg           = vae_cfg,                                           # ← CHANGED (3)
        velocity_net_cfg  = velocity_net_cfg,
        flow_matching_cfg = flow_matching_cfg,
        optimizer_cfg     = optimizer_cfg,
        normalizer        = normalizer,
    )

    # ── Experiment manager ───────────────────────────────────────────────────
    exp_manager = ExperimentManager(
        config      = config,
        model       = model,
        field_names = field_names,
    )
    model.experiment_manager = exp_manager

    # ── Lightning datamodule ──────────────────────────────────────────────────
    dm = WellDataModuleLightning(
        well_base_path    = config.path.well_base_path,
        well_dataset_name = config.path.well_dataset_name,
        batch_size        = train_cfg.batch_size,
        n_steps_input     = train_cfg.n_steps_input,
        n_steps_output    = train_cfg.n_steps_output,
        data_workers      = config.lightning.datamodules.data_workers,
        train_shuffle     = train_cfg.train_shuffle,
        val_shuffle       = train_cfg.val_shuffle,
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = exp_manager.build_callbacks(config.callbacks.callbacks)

    # ── Loggers ──────────────────────────────────────────────────────────────
    loggers = exp_manager.build_loggers(config.loggers.loggers)

    # ── Trainer ──────────────────────────────────────────────────────────────
    tcfg = config.lightning.trainer
    trainer = pl.Trainer(
        accelerator       = tcfg.accelerator,
        strategy          = tcfg.strategy,
        precision         = tcfg.precision,
        max_epochs        = train_cfg.epochs,
        gradient_clip_val = tcfg.gradient_clip_val,
        log_every_n_steps = tcfg.log_every_n_steps,
        callbacks         = callbacks,
        logger            = loggers,
        num_sanity_val_steps = 0,
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == '__main__':
    main()

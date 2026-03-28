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

from the_well.data.normalization import ZScoreNormalization
from the_well.data import WellDataModule
import pytorch_lightning as pl
import argparse
import fsspec
import munch
import torch
import toml
import yaml
import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── project imports ───────────────────────────────────────────────────────────
from waveletflux.utils.configs import VelocityNetConfig, FlowMatchingConfig, OptimizerConfig, LatentVAEConfig
from waveletflux.utils.augmentations import compose_augmentation
from waveletflux.utils.well_data import WellDataModuleLightning
from waveletflux.modules.latent_fm import LatentFlowModule                     
from waveletflux.utils.experiment import ExperimentManager

def get_rank():
    return int(os.environ.get("RANK", "0"))


RANK     = get_rank()
is_rank0 = (RANK == 0)

print(f"[RANK {RANK}] starting train.py", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(prog='LatentFlow', description='Latent Flow Matching')
parser.add_argument('-c', '--config',    type=str, required=True, help='TOML config file')
parser.add_argument('-n', '--num_nodes', type=int, required=True, help='Number of cluster nodes')
parser.add_argument('-d', '--devices',  type=int, required=True, help='GPUs per node')
args = parser.parse_args()

num_nodes            = args.num_nodes
num_devices_per_node = args.devices
num_devices          = num_nodes * num_devices_per_node

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

config = munch.munchify(toml.load(args.config))

pl.seed_everything(config.random.seed)

lt_train = config.lightning.trainer
torch.set_float32_matmul_precision(lt_train.matmul_precision)

# ─────────────────────────────────────────────────────────────────────────────
# Typed configs  (all TOML → Python translation happens here, once)
# ─────────────────────────────────────────────────────────────────────────────

vae_cfg           = LatentVAEConfig.from_config(config.vae)               
velocity_net_cfg  = VelocityNetConfig.from_config(config.velocity_net)
flow_matching_cfg = FlowMatchingConfig.from_config(config.flow_matching, config.sampling)
optimizer_cfg     = OptimizerConfig.from_config(config.optimizer, config.scheduler, config.train)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

train_cfg = config.train

base_data_module = WellDataModule(
    well_base_path     = config.path.well_base_path,
    well_dataset_name  = config.path.well_dataset_name,
    n_steps_input      = train_cfg.n_steps_input,
    n_steps_output     = train_cfg.n_steps_output,
    max_rollout_steps  = train_cfg.max_rollout_steps,
    batch_size         = train_cfg.batch_size,
    use_normalization  = False,
    normalization_type = None,
    transform          = compose_augmentation(*train_cfg.augmentations),
    data_workers       = config.lightning.datamodules.data_workers,
    world_size         = 1,   # Lightning manages distribution
    rank               = 0,
)

data_module = WellDataModuleLightning(
    base_data_module,
    train_shuffle   = train_cfg.train_shuffle,
    val_shuffle     = train_cfg.val_shuffle,
    test_shuffle    = train_cfg.test_shuffle,
    rollout_shuffle = train_cfg.rollout_shuffle,
    target_dtype    = eval(train_cfg.target_dtype),
)

# Load normalization stats (computed from training data) and create normalizer instance
with fsspec.open(config.path.stats_path, mode='r') as f:
    stats = yaml.safe_load(f)

normalizer = ZScoreNormalization(
    stats,
    core_field_names          = base_data_module.train_dataset.core_field_names,
    core_constant_field_names = base_data_module.train_dataset.core_constant_field_names,
)

data_shape  = base_data_module.train_dataset.metadata.spatial_resolution
meta_fields = base_data_module.train_dataset.metadata.field_names
field_names = [item for sublist in meta_fields.values() for item in sublist]
in_channels = len(field_names)

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model = LatentFlowModule(                                            
    in_channels       = in_channels,
    field_names       = field_names,
    data_shape        = data_shape,
    n_steps_output    = train_cfg.n_steps_output,
    scale_weights     = config.loss.scale_weights,
    vae_cfg           = vae_cfg,                                          
    velocity_net_cfg  = velocity_net_cfg,
    flow_matching_cfg = flow_matching_cfg,
    optimizer_cfg     = optimizer_cfg,
    normalizer        = normalizer,
)

# ─────────────────────────────────────────────────────────────────────────────
# Experiment manager
# ─────────────────────────────────────────────────────────────────────────────

model.experiment_manager = ExperimentManager(
    base_dir           = config.path.experiments_path,
    experiment_name    = config.main.exp_name,
    overwrite          = False, # config.resume.ckpt is not None
    loggers            = config.loggers.loggers,
    callbacks          = config.callbacks.callbacks,
    log_every_n_epochs = config.loggers.log_every_n_epochs,
)

callbacks = model.experiment_manager.get_callbacks()

# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

gradient_clip_val = lt_train.gradient_clip_val
gradient_clip_val = gradient_clip_val if isinstance(gradient_clip_val, float) else None

trainer = pl.Trainer(
    accelerator          = lt_train.accelerator,
    strategy             = lt_train.strategy,
    devices              = num_devices_per_node,
    num_nodes            = num_nodes,
    precision            = lt_train.precision,
    callbacks            = callbacks,
    max_epochs           = train_cfg.epochs,
    logger               = False,
    gradient_clip_val    = gradient_clip_val,
    log_every_n_steps    = lt_train.log_every_n_steps,
    num_sanity_val_steps = 0,
    limit_train_batches  = lt_train.limit_train_batches,
    limit_val_batches    = lt_train.limit_val_batches,
)

if is_rank0:
    trainer._logger_connector.configure_logger(
        model.experiment_manager.get_loggers()
    )

# ─────────────────────────────────────────────────────────────────────────────
# Rank-0 bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

if is_rank0:
    model.experiment_manager.save_environment_info()
    model.experiment_manager.copy_config_file(src=args.config)

    em = model.experiment_manager
    em.append_readme('')
    em.append_readme('*** Distributed Environment ***\n')
    em.append_readme(f'> Nodes                : {num_nodes}\n')
    em.append_readme(f'> Devices/node         : {num_devices_per_node}\n')
    em.append_readme(f'> Total devices        : {num_devices}\n')
    em.append_readme(f'> Local batch size     : {train_cfg.batch_size}\n')
    em.append_readme(f'> Global batch size    : {train_cfg.batch_size * num_devices}\n')
    em.append_readme('')

    start_time = datetime.datetime.now()
    em.append_readme('*** Training Started ***\n')
    em.append_readme(f'Started: {start_time}\n')

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

# resume checkpoint if specified
if not config.resume.ckpt:
    trainer.fit(model, datamodule=data_module)
else:   
    trainer.fit(model, datamodule=data_module, ckpt_path=config.resume.ckpt)

if is_rank0:
    end_time = datetime.datetime.now()
    em = model.experiment_manager
    em.append_readme('*** Training Completed ***\n')
    em.append_readme(f'Finished: {end_time}\n')
    em.append_readme(f'Total time: {end_time - start_time}\n')

sys.exit(0)
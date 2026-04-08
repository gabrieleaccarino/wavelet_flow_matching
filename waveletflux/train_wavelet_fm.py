import pytorch_lightning as pl
import argparse
import datetime
import torch
import munch
import toml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waveletflux.utils.configs import WaveletConfig, VelocityNetConfig, FlowMatchingConfig, OptimizerConfig
from waveletflux.utils.augmentations import compose_augmentation, compose_transformation
from waveletflux.utils.well_data import WellDataModuleLightning
from waveletflux.modules.wavelet_fm import WaveletFlowModule
from waveletflux.utils.experiment import ExperimentManager

def get_rank():
    return int(os.environ.get("RANK", "0"))


RANK     = get_rank()
is_rank0 = (RANK == 0)

print(f"[RANK {RANK}] starting train.py", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(prog='WaveletFlow', description='Multi-Scale Wavelet Flow Matching')
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

wavelet_cfg       = WaveletConfig.from_config(config.wavelet)
velocity_net_cfg  = VelocityNetConfig.from_config(config.velocity_net)
flow_matching_cfg = FlowMatchingConfig.from_config(config.flow_matching, config.sampling)
optimizer_cfg     = OptimizerConfig.from_config(config.optimizer, config.scheduler, config.train)

assert len(config.loss.scale_weights) == wavelet_cfg.n_scales, (
    f"loss.scale_weights has {len(config.loss.scale_weights)} entries "
    f"but wavelet.params.n_scales = {wavelet_cfg.n_scales}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

train_cfg = config.train

# Stochastic augmentations — train split only
augmentation = (
    compose_augmentation(*train_cfg.augmentations)
    if train_cfg.augmentations else None
)

# Deterministic transformations — all splits
transformation = (
    compose_transformation(*train_cfg.transformations)
    if train_cfg.transformations else None
)

data_module = WellDataModuleLightning(
    well_base_path    = config.path.well_base_path,
    well_dataset      = config.path.well_dataset_name,
    stats_path        = config.path.stats_path,
    n_steps_input     = config.train.n_steps_input,
    n_steps_output    = config.train.n_steps_output,
    max_rollout_steps = config.train.max_rollout_steps,
    batch_size        = config.train.batch_size,
    num_workers       = config.lightning.datamodules.data_workers,
    train_shuffle     = train_cfg.train_shuffle,
    val_shuffle       = train_cfg.val_shuffle,
    test_shuffle      = train_cfg.test_shuffle,
    # rollout_val/test_shuffle not in TOML, default to False
    target_dtype      = eval(train_cfg.target_dtype),
    augmentation      = augmentation,
    transformation    = transformation,
)

data_module.setup()

normalizer  = data_module.normalizer
data_shape  = data_module.train_dataset.metadata.spatial_resolution
meta_fields = data_module.train_dataset.metadata.field_names
field_names = [item for sublist in meta_fields.values() for item in sublist]

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model = WaveletFlowModule(
    in_channels       = len(field_names),
    field_names       = field_names,
    data_shape        = data_shape,
    n_steps_output    = train_cfg.n_steps_output,
    scale_weights     = config.loss.scale_weights,
    wavelet_cfg       = wavelet_cfg,
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
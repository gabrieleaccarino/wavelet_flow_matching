#!/usr/bin/env python

import argparse
import datetime
import os
import sys

import munch
import pytorch_lightning as pl
import toml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waveletflux.modules.neural_ops import NeuralOperatorModule
from waveletflux.utils.augmentations import compose_augmentation, compose_transformation
from waveletflux.utils.configs import NeuralOperatorConfig, OptimizerConfig
from waveletflux.utils.experiment import ExperimentManager
from waveletflux.utils.well_data import WellDataModuleLightning


def get_rank():
    return int(os.environ.get("RANK", "0"))


RANK = get_rank()
is_rank0 = RANK == 0


def _infer_scalar_cond_dim(dataset) -> int:
    sample = dataset[0]
    scalars = sample.get("constant_scalars")
    return int(scalars.numel()) if scalars is not None else 0


parser = argparse.ArgumentParser(
    prog="train_neural_ops.py",
    description="Train neural-operator and diffusion baselines (FNO / TFNO / WNO / WDNO)",
)
parser.add_argument("-c", "--config", type=str, required=True, help="TOML config file")
parser.add_argument("-n", "--num_nodes", type=int, required=True, help="Number of cluster nodes")
parser.add_argument("-d", "--devices", type=int, required=True, help="GPUs per node")
args = parser.parse_args()

num_nodes = args.num_nodes
num_devices_per_node = args.devices
num_devices = num_nodes * num_devices_per_node

config = munch.munchify(toml.load(args.config))

pl.seed_everything(config.random.seed)
lt_train = config.lightning.trainer
torch.set_float32_matmul_precision(lt_train.matmul_precision)

operator_cfg = NeuralOperatorConfig.from_config(config.neural_operator)
optimizer_cfg = OptimizerConfig.from_config(config.optimizer, config.scheduler, config.train)
train_cfg = config.train

augmentation = (
    compose_augmentation(*train_cfg.augmentations)
    if getattr(train_cfg, "augmentations", None)
    else None
)
transformation = (
    compose_transformation(*train_cfg.transformations)
    if getattr(train_cfg, "transformations", None)
    else None
)

data_module = WellDataModuleLightning(
    well_base_path=config.path.well_base_path,
    well_dataset=config.path.well_dataset_name,
    stats_path=config.path.stats_path,
    n_steps_input=train_cfg.n_steps_input,
    n_steps_output=train_cfg.n_steps_output,
    max_rollout_steps=train_cfg.max_rollout_steps,
    batch_size=train_cfg.batch_size,
    num_workers=config.lightning.datamodules.data_workers,
    train_shuffle=train_cfg.train_shuffle,
    val_shuffle=train_cfg.val_shuffle,
    test_shuffle=train_cfg.test_shuffle,
    target_dtype=eval(train_cfg.target_dtype),
    augmentation=augmentation,
    transformation=transformation,
)

data_module.setup()

normalizer = data_module.normalizer
data_shape = data_module.train_dataset.metadata.spatial_resolution
meta_fields = data_module.train_dataset.metadata.field_names
field_names = [item for sublist in meta_fields.values() for item in sublist]
scalar_cond_dim = _infer_scalar_cond_dim(data_module.train_dataset)

model = NeuralOperatorModule(
    operator_cfg=operator_cfg,
    n_fields=len(field_names),
    field_names=field_names,
    data_shape=data_shape,
    n_steps_input=train_cfg.n_steps_input,
    n_steps_output=train_cfg.n_steps_output,
    scalar_cond_dim=scalar_cond_dim,
    optimizer_cfg=optimizer_cfg,
    normalizer=normalizer,
)

model.experiment_manager = ExperimentManager(
    base_dir=config.path.experiments_path,
    experiment_name=config.main.exp_name,
    overwrite=False,
    loggers=config.loggers.loggers,
    callbacks=config.callbacks.callbacks,
    log_every_n_epochs=config.loggers.log_every_n_epochs,
)

callbacks = model.experiment_manager.get_callbacks()

gradient_clip_val = lt_train.gradient_clip_val
gradient_clip_val = gradient_clip_val if isinstance(gradient_clip_val, float) else None

trainer = pl.Trainer(
    accelerator=lt_train.accelerator,
    strategy=lt_train.strategy,
    devices=num_devices_per_node,
    num_nodes=num_nodes,
    precision=lt_train.precision,
    callbacks=callbacks,
    max_epochs=train_cfg.epochs,
    logger=False,
    gradient_clip_val=gradient_clip_val,
    log_every_n_steps=lt_train.log_every_n_steps,
    num_sanity_val_steps=0,
    limit_train_batches=getattr(lt_train, "limit_train_batches", 1.0),
    limit_val_batches=getattr(lt_train, "limit_val_batches", 1.0),
)

if is_rank0:
    trainer._logger_connector.configure_logger(model.experiment_manager.get_loggers())
    model.experiment_manager.save_environment_info()
    model.experiment_manager.copy_config_file(src=args.config)

    em = model.experiment_manager
    em.append_readme("")
    em.append_readme("*** Distributed Environment ***\n")
    em.append_readme(f"> Nodes                : {num_nodes}\n")
    em.append_readme(f"> Devices/node         : {num_devices_per_node}\n")
    em.append_readme(f"> Total devices        : {num_devices}\n")
    em.append_readme(f"> Local batch size     : {train_cfg.batch_size}\n")
    em.append_readme(f"> Global batch size    : {train_cfg.batch_size * num_devices}\n")
    em.append_readme(f"> Model                : {operator_cfg.model_type.upper()}\n")
    em.append_readme(f"> Fields               : {len(field_names)}\n")
    em.append_readme(f"> Scalar cond dim      : {scalar_cond_dim}\n")
    em.append_readme(f"> Spatial shape        : {tuple(data_shape)}\n")
    em.append_readme("")

    start_time = datetime.datetime.now()
    em.append_readme("*** Training Started ***\n")
    em.append_readme(f"Started: {start_time}\n")

if not getattr(config, "resume", None) or not config.resume.ckpt:
    trainer.fit(model, datamodule=data_module)
else:
    trainer.fit(model, datamodule=data_module, ckpt_path=config.resume.ckpt)

if is_rank0:
    end_time = datetime.datetime.now()
    em = model.experiment_manager
    em.append_readme("*** Training Completed ***\n")
    em.append_readme(f"Finished: {end_time}\n")
    em.append_readme(f"Total time: {end_time - start_time}\n")

sys.exit(0)

#from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.callbacks import Callback

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import mlflow
import torch
import time
import csv
import os
import shutil

class EmergencySaveCallback(pl.Callback):
    """
    Saves an emergency checkpoint if training crashes or is interrupted.
    Useful in long HPC runs or environments with preemption.
    """
    def on_exception(self, trainer, pl_module, exception):
        # Build safe emergency checkpoint path
        ckpt_dir = pl_module.experiment_manager.dirs["checkpoints"]
        emergency_path = os.path.join(ckpt_dir, f"emergency_ckpt_epoch_{trainer.current_epoch}.ckpt")

        trainer.save_checkpoint(emergency_path)
        print(f"\n > Emergency checkpoint saved at: {emergency_path}\n")


class PerformanceTrackingCallback(pl.Callback):
    """
    Logs training performance, timing, throughput, GPU usage, model size,
    and parameter counts. Logs to:
        - MLflow (if active)
        - local experiment folder (performance/)
    """

    def __init__(self, perf_dir: str):
        super().__init__()

        self.perf_dir = perf_dir
        os.makedirs(self.perf_dir, exist_ok=True)

        # Internal timers
        self._epoch_start = None
        self._step_start = None

        # Offline CSV files
        self.epoch_csv = os.path.join(self.perf_dir, "epoch_times.csv")
        self.step_csv = os.path.join(self.perf_dir, "step_times.csv")
        self.gpu_csv = os.path.join(self.perf_dir, "gpu_memory.csv")
        self.model_info_json = os.path.join(self.perf_dir, "model_info.json")

        # Initialize CSV headers
        with open(self.epoch_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "epoch_time_sec", "samples_per_sec"])

        with open(self.step_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "step_time_sec", "samples_per_sec"])

        with open(self.gpu_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "gpu_memory_gb"])

    # ------------------------------------------------------------
    # At the beginning of training, record model metadata
    # ------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):

        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }

        import json
        with open(self.model_info_json, "w") as f:
            json.dump(model_info, f, indent=4)

        # Log to MLflow if available
        if isinstance(trainer.logger, pl.loggers.MLFlowLogger):
            mlflow.log_params(model_info)

    # ------------------------------------------------------------
    # Step-level timing
    # ------------------------------------------------------------
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_start = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # store batch size used for this step
        self._last_batch_size = batch["input_fields"].shape[0]

        step_time = time.time() - self._step_start
        B = self._last_batch_size
        samples_per_sec = B / step_time

        # CSV
        with open(self.step_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trainer.global_step, step_time, samples_per_sec])

        # MLflow
        if isinstance(trainer.logger, pl.loggers.MLFlowLogger):
            mlflow.log_metric("perf/step_time", step_time, step=trainer.global_step)
            mlflow.log_metric("perf/samples_per_sec", samples_per_sec, step=trainer.global_step)

        # GPU
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**3
            with open(self.gpu_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([trainer.global_step, mem])
            if isinstance(trainer.logger, pl.loggers.MLFlowLogger):
                mlflow.log_metric("gpu/peak_memory_gb", mem, step=trainer.global_step)

    # ------------------------------------------------------------
    # Epoch-level timing
    # ------------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):

        epoch_time = time.time() - self._epoch_start
        epoch = trainer.current_epoch

        data_loader = trainer.train_dataloader
        if isinstance(data_loader, list):
            # we are in a distributed environment
            data_loader = data_loader[0]

        B = getattr(data_loader, "batch_size", None)
        if B is None:
            # fallback: use last seen batch size
            B = getattr(self, "_last_batch_size", 1)

        # Compute throughput
        total_steps = trainer.num_training_batches
        samples_per_sec = (total_steps * B) / epoch_time

        # CSV
        with open(self.epoch_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_time, samples_per_sec])

        # MLFlow
        if isinstance(trainer.logger, pl.loggers.MLFlowLogger):
            mlflow.log_metric("perf/epoch_time", epoch_time, step=epoch)
            mlflow.log_metric("perf/samples_per_sec_epoch", samples_per_sec, step=epoch)


#class PortableCheckpointCallback(Callback):
#    """
#    Saves a self-contained, FP32, single-file checkpoint after every
#    `every_n_epochs` epochs. Works with DeepSpeed ZeRO (any stage),
#    DDP, and single-GPU training — no external config needed to reload.
#
#    Load with:
#        WaveletFlowModule.from_checkpoint("portable_epoch0010.ckpt")
#    """
#
#    def __init__(self, dirpath: str, every_n_epochs: int = 1):
#        self.dirpath         = dirpath
#        self.every_n_epochs  = every_n_epochs
#
#    def on_train_epoch_end(self, trainer, pl_module):
#        epoch = trainer.current_epoch
#        if epoch % self.every_n_epochs != 0:
#            return
#        # Barrier: all ranks must finish the epoch before rank-0 consolidates
#        trainer.strategy.barrier()
#        if trainer.global_rank != 0:
#            return
#
#        out_path = os.path.join(self.dirpath, f"portable_epoch{epoch:04d}.ckpt")
#        os.makedirs(self.dirpath, exist_ok=True)
#
#        # ── DeepSpeed: consolidate shards → FP32 ────────────────────────────
#        try:
#            from deepspeed.utils.zero_to_fp32 import (
#                get_fp32_state_dict_from_zero_checkpoint,
#            )
#            ds_dir  = trainer.checkpoint_callback.dirpath
#            state_dict = get_fp32_state_dict_from_zero_checkpoint(ds_dir)
#        except Exception:
#            # DDP / single-GPU: Lightning already gives a full state dict on rank 0
#            state_dict = pl_module.state_dict()
#
#        torch.save(
#            {
#                "state_dict":       state_dict,
#                "hyper_parameters": dict(pl_module.hparams),
#                "epoch":            epoch,
#                "global_step":      trainer.global_step,
#            },
#            out_path,
#        )
#        print(f"[rank0] portable checkpoint → {out_path}", flush=True)


class PortableModelCheckpoint(ModelCheckpoint):
    """
    Drop-in replacement for ModelCheckpoint that saves an additional
    self-contained FP32 portable checkpoint alongside every native .ckpt.

    The portable file mirrors ModelCheckpoint exactly:
      - same save_top_k / monitor / save_last logic
      - same filename, with a _portable.ckpt suffix
      - works with DeepSpeed ZeRO (any stage), DDP, and single-GPU

    Load with:
        WaveletFlowModule.from_checkpoint("run-epoch=5-val_loss=0.0312_portable.ckpt")
    """

    def _save_checkpoint(self, trainer, filepath: str):
        # 1. Let ModelCheckpoint do its normal save (native format, all ranks)
        super()._save_checkpoint(trainer, filepath)

        # 2. All ranks must finish writing shards before rank-0 consolidates
        trainer.strategy.barrier()

        if trainer.global_rank != 0:
            return

        # 3. Derive portable path from the native checkpoint path
        #    filepath may be a directory (DeepSpeed) or a .ckpt file (DDP/single-GPU)
        base     = filepath.rstrip("/").rstrip("\\")
        portable = base.replace(".ckpt", "") + "_portable.ckpt"

        self._write_portable(trainer, filepath, portable)
        self._maybe_update_best_aliases(filepath, portable)

    def _remove_checkpoint(self, trainer, filepath: str):
        # Mirror deletions: when ModelCheckpoint prunes a stale checkpoint,
        # prune the corresponding portable file too
        super()._remove_checkpoint(trainer, filepath)

        if trainer.global_rank != 0:
            return

        portable = filepath.replace(".ckpt", "") + "_portable.ckpt"
        if os.path.exists(portable):
            os.remove(portable)

    # ------------------------------------------------------------------ #

    def _write_portable(self, trainer, source_path: str, out_path: str):
        pl_module = trainer.lightning_module

        # ── DeepSpeed: consolidate shards → FP32 ────────────────────────
        try:
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )
            # With DeepSpeed, source_path is a directory containing the shards
            state_dict = get_fp32_state_dict_from_zero_checkpoint(source_path)
        except Exception:
            # DDP / single-GPU: Lightning already gives a full state dict on rank 0
            state_dict = pl_module.state_dict()

        payload = {
            "state_dict":       state_dict,
            "hyper_parameters": dict(pl_module.hparams),
            "epoch":            trainer.current_epoch,
            "global_step":      trainer.global_step,
        }
        if hasattr(pl_module, "_export_ema_checkpoint_state"):
            payload.update(pl_module._export_ema_checkpoint_state())

        torch.save(payload, out_path)
        #print(f"[rank0] portable checkpoint → {out_path}", flush=True)

    def _maybe_update_best_aliases(self, filepath: str, portable_path: str):
        best_model_path = getattr(self, "best_model_path", "")
        if not best_model_path:
            return

        if os.path.abspath(filepath) != os.path.abspath(best_model_path):
            return

        ckpt_dir = os.path.dirname(filepath.rstrip("/").rstrip("\\"))
        best_native = os.path.join(ckpt_dir, "best.ckpt")
        best_portable = os.path.join(ckpt_dir, "best_portable.ckpt")

        self._copy_checkpoint_artifact(filepath, best_native)
        self._copy_checkpoint_artifact(portable_path, best_portable)

    def _copy_checkpoint_artifact(self, source_path: str, target_path: str):
        if os.path.abspath(source_path) == os.path.abspath(target_path):
            return

        if os.path.isdir(target_path):
            shutil.rmtree(target_path)
        elif os.path.exists(target_path):
            os.remove(target_path)

        if os.path.isdir(source_path):
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)

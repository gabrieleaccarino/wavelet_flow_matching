from contextlib import contextmanager
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from waveletflux.baselines import WDNO2d, WNO2d
from waveletflux.utils.configs import NeuralOperatorConfig, OptimizerConfig


class NeuralOperatorModule(pl.LightningModule):
    """LightningModule for neural-operator baselines."""

    def __init__(
        self,
        operator_cfg: NeuralOperatorConfig,
        n_fields: int,
        field_names: list[str],
        data_shape: tuple[int, int],
        n_steps_input: int,
        n_steps_output: int,
        scalar_cond_dim: int,
        optimizer_cfg: OptimizerConfig,
        normalizer: Any,
    ):
        super().__init__()

        self.operator_cfg = operator_cfg
        self.model_type = operator_cfg.model_type.lower()
        self.n_fields = n_fields
        self.field_names = field_names
        self.data_shape = tuple(data_shape)
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.scalar_cond_dim = scalar_cond_dim
        self.optimizer_cfg = optimizer_cfg
        self.normalizer = normalizer
        self.experiment_manager = None
        self.use_ema = self.model_type == "wdno" and operator_cfg.ema_decay > 0.0
        self.ema_decay = float(operator_cfg.ema_decay)
        self.ema_update_every = max(int(operator_cfg.ema_update_every), 1)
        self.ema_start_step = int(operator_cfg.ema_start_step)
        self.register_buffer("ema_num_updates", torch.zeros((), dtype=torch.long), persistent=True)
        self._ema_state = None
        self._ema_backup_state = None
        self._ema_active = False
        self.wdno_eval_ensemble_size = 4
        self.wdno_eval_base_seed = 0

        in_channels = n_fields * n_steps_input
        out_channels = n_fields * n_steps_output

        if self.model_type == "wno":
            self.model = WNO2d(
                n_fields=n_fields,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                size=self.data_shape,
                hidden_channels=operator_cfg.hidden_channels,
                n_layers=operator_cfg.n_layers,
                level=operator_cfg.level,
                wavelet=operator_cfg.wavelet,
                mode=operator_cfg.mode,
                scalar_cond_dim=scalar_cond_dim,
                grid_range=operator_cfg.grid_range,
                padding=operator_cfg.padding,
                fc_channels=operator_cfg.fc_channels,
            )
        elif self.model_type == "wdno":
            self.model = WDNO2d(
                n_fields=n_fields,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                size=self.data_shape,
                hidden_channels=operator_cfg.hidden_channels,
                n_res=operator_cfg.n_res,
                n_levels=operator_cfg.n_levels,
                n_scales=operator_cfg.n_scales,
                cond_dim=operator_cfg.cond_dim,
                wavelet=operator_cfg.wavelet,
                mode=operator_cfg.mode,
                scalar_cond_dim=scalar_cond_dim,
                diffusion_steps=operator_cfg.diffusion_steps,
                sampling_steps=operator_cfg.sampling_steps,
                beta_schedule=operator_cfg.beta_schedule,
                ddim_eta=operator_cfg.ddim_eta,
            )
        elif self.model_type in {"fno", "tfno"}:
            try:
                from neuralop.models import FNO, TFNO
            except ImportError as exc:
                raise ImportError(
                    f"{self.model_type.upper()} requires `neuralop` to be installed."
                ) from exc

            if self.model_type == "fno":
                self.model = FNO(
                    n_modes=(operator_cfg.modes, operator_cfg.modes),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=operator_cfg.hidden_channels,
                    n_layers=operator_cfg.n_layers,
                )
            else:
                self.model = TFNO(
                    n_modes=(operator_cfg.modes, operator_cfg.modes),
                    in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=operator_cfg.hidden_channels,
                    n_layers=operator_cfg.n_layers,
                    factorization="tucker",
                    rank=operator_cfg.tucker_rank,
                )
        else:
            raise ValueError(f"Unknown operator type: {operator_cfg.model_type}")

        if self.use_ema:
            self._sync_ema_state_from_model()

        self.save_hyperparameters(
            {
                "operator_cfg": operator_cfg.to_dict(),
                "n_fields": n_fields,
                "field_names": field_names,
                "data_shape": self.data_shape,
                "n_steps_input": n_steps_input,
                "n_steps_output": n_steps_output,
                "scalar_cond_dim": scalar_cond_dim,
                "optimizer_cfg": optimizer_cfg.to_dict(),
                "normalizer_stats": {
                    "mean": {k: v.tolist() for k, v in normalizer.means.items()},
                    "std": {k: v.tolist() for k, v in normalizer.stds.items()},
                    "mean_delta": {
                        k: v.tolist() for k, v in normalizer.means_delta.items()
                    },
                    "std_delta": {
                        k: v.tolist() for k, v in normalizer.stds_delta.items()
                    },
                },
                "normalizer_core_field_names": normalizer.core_field_names,
                "normalizer_core_constant_field_names": normalizer.core_constant_field_names,
            }
        )

    def _get_scalar_conds(self, batch) -> Optional[torch.Tensor]:
        scalars = batch.get("constant_scalars")
        scalars = scalars.to(self.device) if scalars is not None else None
        n_scalars = int(scalars.shape[-1]) if scalars is not None else 0

        time_grid = batch.get("input_time_grid")
        dt = None
        if time_grid is not None and time_grid.shape[-1] >= 2:
            dt = (time_grid[:, 1] - time_grid[:, 0]).unsqueeze(-1).to(self.device)

        if self.scalar_cond_dim == 0:
            return None

        parts = []
        if scalars is not None:
            parts.append(scalars)

        if dt is not None and self.scalar_cond_dim == n_scalars + 1:
            parts.append(dt)
        elif scalars is None and dt is not None and self.scalar_cond_dim == 1:
            parts.append(dt)

        if not parts:
            return None

        scalar_conds = torch.cat(parts, dim=-1)
        if scalar_conds.shape[-1] != self.scalar_cond_dim:
            raise ValueError(
                f"Expected {self.scalar_cond_dim} scalar conditions, got "
                f"{scalar_conds.shape[-1]}. Check checkpoint/config conditioning dims."
            )
        return scalar_conds

    def _clone_model_state(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _ensure_ema_state_device(self) -> None:
        if not self.use_ema or self._ema_state is None:
            return
        model_state = self.model.state_dict()
        for name, tensor in model_state.items():
            ema_tensor = self._ema_state.get(name)
            if ema_tensor is None or ema_tensor.shape != tensor.shape:
                self._ema_state[name] = tensor.detach().clone()
                continue
            if ema_tensor.device != tensor.device or ema_tensor.dtype != tensor.dtype:
                self._ema_state[name] = ema_tensor.to(device=tensor.device, dtype=tensor.dtype)

    def _sync_ema_state_from_model(self) -> None:
        if not self.use_ema:
            return
        self._ema_state = self._clone_model_state()

    @torch.no_grad()
    def _update_ema_state(self) -> None:
        if not self.use_ema:
            return
        self._ensure_ema_state_device()

        model_state = self.model.state_dict()
        if int(self.ema_num_updates.item()) == 0 or self._ema_state is None:
            self._sync_ema_state_from_model()
        else:
            for name, tensor in model_state.items():
                ema_tensor = self._ema_state[name]
                tensor = tensor.detach()
                if torch.is_floating_point(tensor):
                    ema_tensor.mul_(self.ema_decay).add_(tensor, alpha=1.0 - self.ema_decay)
                else:
                    ema_tensor.copy_(tensor)

        self.ema_num_updates.add_(1)

    def _apply_ema_weights(self) -> bool:
        if (
            not self.use_ema
            or self._ema_active
            or self._ema_state is None
            or int(self.ema_num_updates.item()) == 0
        ):
            return False

        self._ensure_ema_state_device()
        self._ema_backup_state = self._clone_model_state()
        self.model.load_state_dict(self._ema_state, strict=True)
        self._ema_active = True
        return True

    def _restore_model_weights(self) -> None:
        if not self._ema_active or self._ema_backup_state is None:
            return
        self.model.load_state_dict(self._ema_backup_state, strict=True)
        self._ema_backup_state = None
        self._ema_active = False

    @contextmanager
    def _ema_scope(self, enabled: bool = True):
        applied = self._apply_ema_weights() if enabled else False
        try:
            yield
        finally:
            if applied:
                self._restore_model_weights()

    def _export_ema_checkpoint_state(self) -> dict:
        if not self.use_ema or self._ema_state is None:
            return {}
        return {
            "ema_state_dict": {k: v.detach().cpu() for k, v in self._ema_state.items()},
            "ema_num_updates": self.ema_num_updates.detach().cpu(),
        }

    def _load_ema_checkpoint_state(self, checkpoint: dict) -> None:
        if not self.use_ema:
            return

        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is None:
            self._sync_ema_state_from_model()
            self.ema_num_updates.zero_()
            return

        self._ema_state = {k: v.detach().clone() for k, v in ema_state.items()}
        self._ensure_ema_state_device()

        ema_num_updates = checkpoint.get("ema_num_updates")
        if ema_num_updates is None:
            self.ema_num_updates.zero_()
        else:
            self.ema_num_updates.copy_(
                torch.as_tensor(
                    ema_num_updates,
                    dtype=self.ema_num_updates.dtype,
                    device=self.ema_num_updates.device,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model_type in {"wno", "wdno"}:
            return self.model(x, scalar_conds=scalar_conds)
        return self.model(x)

    def _prepare_supervised_batch(self, batch):
        x = self.normalizer.normalize_flattened(
            batch["input_fields"].to(self.device), mode="variable"
        )
        y = self.normalizer.normalize_flattened(
            batch["output_fields"].to(self.device), mode="variable"
        )
        scalar_conds = self._get_scalar_conds(batch)

        x = rearrange(x, "B Ti H W F -> B (Ti F) H W")
        y = rearrange(y, "B To H W F -> B (To F) H W")
        return x, y, scalar_conds

    @torch.no_grad()
    def calibrate_wdno_coeff_scales(
        self,
        dataset,
        batch_size: int,
        num_batches: int = 32,
    ) -> None:
        if self.model_type != "wdno":
            return
        if num_batches < 1:
            return

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: torch.utils.data.dataloader.default_collate(batch),
        )

        coeff_sumsq = {
            i: torch.zeros_like(getattr(self.model, f"coeff_scale_{i}"))
            for i in self.model.scale_shapes
        }
        coeff_count = 0

        was_training = self.training
        self.eval()
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= num_batches:
                break

            x, y, scalar_conds = self._prepare_supervised_batch(batch)
            if hasattr(self.model, "_coeff_dict_from_supervised_tensors"):
                coeffs = self.model._coeff_dict_from_supervised_tensors(x, y)
            else:
                y_fields = rearrange(
                    y,
                    "B (To F) H W -> B To H W F",
                    To=self.n_steps_output,
                    F=self.n_fields,
                )
                coeffs = self.model._coeff_dict_from_fields(y_fields)
            for i, coeff in coeffs.items():
                coeff_sumsq[i] += coeff.square().sum(dim=(0, 2, 3), keepdim=True)
            ref = next(iter(coeffs.values()))
            coeff_count += ref.shape[0] * ref.shape[2] * ref.shape[3]

        coeff_scales = {
            i: (coeff_sumsq[i] / max(coeff_count, 1)).sqrt().mul(3.0).clamp(min=1.0)
            for i in coeff_sumsq
        }
        self.model.set_coeff_scales(coeff_scales)
        if self.use_ema:
            self._sync_ema_state_from_model()
        if was_training:
            self.train()

    def _shared_step(self, batch, log_prefix: str) -> torch.Tensor:
        x, y, scalar_conds = self._prepare_supervised_batch(batch)
        if self.model_type == "wdno":
            loss = self.model.loss(x, y, scalar_conds=scalar_conds)
        else:
            loss = (self(x, scalar_conds=scalar_conds) - y).square().mean()
        self.log(
            f"{log_prefix}_loss",
            loss,
            on_step=(log_prefix == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_fields"].shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    @torch.no_grad()
    def sample(self, batch, batch_idx):
        with self._ema_scope(enabled=self.use_ema):
            x, _, scalar_conds = self._prepare_supervised_batch(batch)
            yhat_norm = self(x, scalar_conds=scalar_conds)
            yhat_norm = rearrange(
                yhat_norm,
                "B (To F) H W -> B To H W F",
                To=self.n_steps_output,
                F=self.n_fields,
            )
            yhat = self.normalizer.denormalize_flattened(yhat_norm, mode="variable")
            y = batch["output_fields"].to(self.device)
            return yhat, y

    @torch.no_grad()
    def _wdno_eval_sample(self, batch, num_samples: Optional[int] = None):
        if self.model_type != "wdno":
            raise RuntimeError("_wdno_eval_sample() is only valid for WDNO models.")

        num_samples = int(num_samples or self.wdno_eval_ensemble_size)
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        preds = []
        try:
            for seed_offset in range(num_samples):
                torch.manual_seed(self.wdno_eval_base_seed + seed_offset)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.wdno_eval_base_seed + seed_offset)
                yhat, y = self.sample(batch, batch_idx=0)
                preds.append(yhat)
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        yhat_ens = torch.stack(preds, dim=1)
        return yhat_ens.mean(dim=1), y

    @torch.no_grad()
    def _wdno_eval_loss(self, batch, num_samples: Optional[int] = None) -> torch.Tensor:
        yhat, y = self._wdno_eval_sample(batch, num_samples=num_samples)
        yhat_norm = self.normalizer.normalize_flattened(yhat, mode="variable")
        y_norm = self.normalizer.normalize_flattened(y, mode="variable")
        return (yhat_norm - y_norm).square().mean()

    @torch.no_grad()
    def rollout(
        self,
        seed: torch.Tensor,
        scalar_conds: Optional[torch.Tensor],
        rollout_steps: int,
        normalized_input: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        with self._ema_scope(enabled=self.use_ema):
            if self.model_type == "wno":
                return self.model.rollout(
                    seed=seed,
                    scalar_conds=scalar_conds,
                    rollout_steps=rollout_steps,
                    normalizer=self.normalizer,
                    normalized_input=normalized_input,
                    verbose=verbose,
                )

            if seed.ndim != 5:
                raise ValueError(
                    f"Expected seed shape `(B, Ti, H, W, F)`, got {tuple(seed.shape)}."
                )
            if rollout_steps < 1:
                raise ValueError(f"`rollout_steps` must be >= 1, got {rollout_steps}.")

            device = self.device
            dtype = next(self.parameters()).dtype
            seed = seed.to(device=device, dtype=dtype)

            if not normalized_input:
                windows_phys = seed[:, -self.n_steps_input :].clone()
                windows_norm = None
            else:
                windows_norm = seed[:, -self.n_steps_input :].clone()
                windows_phys = None

            scalar_conds = (
                scalar_conds.to(device=device, dtype=dtype) if scalar_conds is not None else None
            )

            batch_size, _, height, width, _ = seed.shape
            preds = torch.zeros(
                batch_size,
                rollout_steps,
                height,
                width,
                self.n_fields,
                device="cpu",
                dtype=torch.float32,
            )

            step_out = 0
            while step_out < rollout_steps:
                if windows_phys is not None:
                    window_norm = self.normalizer.normalize_flattened(
                        windows_phys, mode="variable"
                    )
                else:
                    window_norm = windows_norm

                x_in = rearrange(window_norm, "B Ti H W F -> B (Ti F) H W")
                yhat = self(x_in, scalar_conds=scalar_conds)
                yhat_norm = rearrange(
                    yhat,
                    "B (To F) H W -> B To H W F",
                    To=self.n_steps_output,
                    F=self.n_fields,
                )
                yhat_phys = self.normalizer.denormalize_flattened(yhat_norm, mode="variable")

                frames_to_store = min(self.n_steps_output, rollout_steps - step_out)
                preds[:, step_out : step_out + frames_to_store] = (
                    yhat_phys[:, :frames_to_store].detach().float().cpu()
                )
                step_out += frames_to_store

                if verbose:
                    print(
                        f"  rollout frames {step_out}/{rollout_steps}",
                        end="\r",
                    )

                if windows_phys is not None:
                    windows_phys = torch.cat([windows_phys, yhat_phys], dim=1)[
                        :, -self.n_steps_input :
                    ]
                else:
                    windows_norm = torch.cat([windows_norm, yhat_norm], dim=1)[
                        :, -self.n_steps_input :
                    ]

            if verbose:
                print()
            return preds

    def configure_optimizers(self):
        oc = self.optimizer_cfg
        optimizer = getattr(torch.optim, oc.cls)(
            (p for p in self.parameters() if p.requires_grad),
            **oc.optimizer_kwargs(),
        )

        if not oc.scheduler_cls:
            return optimizer

        cosine_epochs = max(self.trainer.max_epochs - oc.warmup_epochs, 1)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=oc.warmup_epochs,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_epochs,
                    eta_min=oc.eta_min,
                ),
            ],
            milestones=[oc.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def on_train_epoch_end(self):
        if self.experiment_manager is None:
            return

        should_save = (
            self.current_epoch % self.experiment_manager.log_every_n_epochs == 0
            or self.current_epoch == self.trainer.max_epochs - 1
        )
        if not should_save:
            return

        if self.model_type == "wdno":
            batch, batch_idx = self._get_validation_sample()
            yhat, y = self._wdno_eval_sample(batch)
        else:
            batch, batch_idx = self._get_random_validation_sample()
            yhat, y = self.sample(batch, batch_idx)
        self.experiment_manager.dump_sample_plots(
            yhat=rearrange(yhat, "B T H W C -> B C T H W"),
            y=rearrange(y, "B T H W C -> B C T H W"),
            field_names=self.field_names,
            fig_name=f"epoch_{self.current_epoch}_val_batch_{batch_idx}.png",
            cmap="RdBu_r",
            dpi=300,
        )

    def on_before_zero_grad(self, optimizer):
        if not self.use_ema:
            return

        step = int(self.global_step)
        if step < self.ema_start_step:
            self._sync_ema_state_from_model()
            return

        if step % self.ema_update_every != 0:
            return

        self._update_ema_state()

    def on_validation_start(self):
        self._apply_ema_weights()

    def on_validation_epoch_end(self):
        if self.model_type != "wdno":
            return
        batch, _ = self._get_validation_sample()
        sample_loss = self._wdno_eval_loss(batch)
        self.log(
            "val_sample_loss",
            sample_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["output_fields"].shape[0],
        )

    def on_validation_end(self):
        self._restore_model_weights()

    def on_test_start(self):
        self._apply_ema_weights()

    def on_test_end(self):
        self._restore_model_weights()

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(self._export_ema_checkpoint_state())

    def on_load_checkpoint(self, checkpoint):
        self._load_ema_checkpoint_state(checkpoint)

    def _get_random_validation_sample(self):
        loader = self.trainer.val_dataloaders
        if isinstance(loader, list):
            loader = loader[0]
        idx = torch.randint(0, len(loader), (1,)).item()
        for i, batch in enumerate(loader):
            if i == idx:
                break
        batch_size = batch["output_fields"].shape[0]
        sample_idx = torch.randint(0, batch_size, (1,)).item()
        return {k: v[sample_idx : sample_idx + 1] for k, v in batch.items()}, idx

    def _get_validation_sample(self, batch_idx: int = 0, sample_idx: int = 0):
        loader = self.trainer.val_dataloaders
        if isinstance(loader, list):
            loader = loader[0]
        for i, batch in enumerate(loader):
            if i == batch_idx:
                return {k: v[sample_idx : sample_idx + 1] for k, v in batch.items()}, batch_idx
        raise IndexError(f"Validation batch index {batch_idx} is out of range.")

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        map_location="cpu",
        stats: dict = None,
    ) -> "NeuralOperatorModule":
        from the_well.data.normalization import ZScoreNormalization

        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hp = ckpt["hyper_parameters"]

        if stats is not None:
            normalizer_stats = stats
        elif hp.get("normalizer_stats") is not None:
            normalizer_stats = hp["normalizer_stats"]
        elif "normalizer_means" in hp and "normalizer_stds" in hp:
            zeros = {k: 0.0 for k in hp["normalizer_means"]}
            normalizer_stats = {
                "mean": hp["normalizer_means"],
                "std": hp["normalizer_stds"],
                "mean_delta": zeros,
                "std_delta": zeros,
            }
        else:
            raise KeyError(
                f"Cannot find normalizer stats in checkpoint. Keys: {list(hp.keys())}"
            )

        normalizer = ZScoreNormalization(
            stats=normalizer_stats,
            core_field_names=hp["normalizer_core_field_names"],
            core_constant_field_names=hp["normalizer_core_constant_field_names"],
        )

        model = cls(
            operator_cfg=NeuralOperatorConfig.from_dict(hp["operator_cfg"]),
            n_fields=hp["n_fields"],
            field_names=hp["field_names"],
            data_shape=tuple(hp["data_shape"]),
            n_steps_input=hp["n_steps_input"],
            n_steps_output=hp["n_steps_output"],
            scalar_cond_dim=hp["scalar_cond_dim"],
            optimizer_cfg=OptimizerConfig.from_dict(hp["optimizer_cfg"]),
            normalizer=normalizer,
        )
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["state_dict"], strict=False)
        allowed_missing = {"ema_num_updates"}
        if model.model_type == "wdno":
            allowed_missing.update(
                f"model.coeff_scale_{i}" for i in model.model.scale_shapes
            )

        unexpected = [key for key in unexpected_keys if key not in allowed_missing]
        missing = [key for key in missing_keys if key not in allowed_missing]
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint state_dict is incompatible. "
                f"Missing keys: {missing}. Unexpected keys: {unexpected}."
            )
        model._load_ema_checkpoint_state(ckpt)
        return model

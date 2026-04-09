from typing import Any, Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
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
        if scalars is None:
            return None
        return scalars.to(self.device)

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
    def rollout(
        self,
        seed: torch.Tensor,
        scalar_conds: Optional[torch.Tensor],
        rollout_steps: int,
        normalized_input: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
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
        optimizer = getattr(torch.optim, oc.cls)(self.parameters(), **oc.optimizer_kwargs())

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

    def _get_random_validation_sample(self):
        loader = self.trainer.val_dataloaders
        idx = torch.randint(0, len(loader), (1,)).item()
        for i, batch in enumerate(loader):
            if i == idx:
                break
        batch_size = batch["output_fields"].shape[0]
        sample_idx = torch.randint(0, batch_size, (1,)).item()
        return {k: v[sample_idx : sample_idx + 1] for k, v in batch.items()}, idx

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
        model.load_state_dict(ckpt["state_dict"])
        return model

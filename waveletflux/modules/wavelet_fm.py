from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from typing import Any, Dict, List
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from torch.optim import *
import torch
import math

from waveletflux.wavelets.DWT2D import WaveletDecomposer2D
from waveletflux.models.wavelet_unet import WaveletFlowUNet
from waveletflux.utils.configs import (
    WaveletConfig,
    VelocityNetConfig,
    FlowMatchingConfig,
    OptimizerConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _subband_vmse(
    pred:             torch.Tensor,
    target:           torch.Tensor,
    in_channels:      int,
    n_steps_output:   int,
    n_coeff:          int,
    subband_weights:  torch.Tensor | None = None,   # ← missing in doc 3
) -> torch.Tensor:
    if n_coeff == 1:
        var = target.var(dim=(2, 3), keepdim=True).detach().clamp(min=1e-4)
        return F.mse_loss(pred / var.sqrt(), target / var.sqrt())

    pred_r = rearrange(pred,   'B (C T K) H W -> (B C T) K H W', C=in_channels, T=n_steps_output, K=n_coeff)
    tgt_r  = rearrange(target, 'B (C T K) H W -> (B C T) K H W', C=in_channels, T=n_steps_output, K=n_coeff)

    mse_k = (pred_r - tgt_r).square().mean(dim=(-2, -1))      # (BCT, K)
    var_k = tgt_r.var(dim=(-2, -1)).detach().clamp(min=1e-4)  # (BCT, K)
    ratio = mse_k / var_k                                     # (BCT, K)

    if subband_weights is not None:
        w = subband_weights[:n_coeff].to(ratio)
        w = w / w.sum()
        return (ratio * w).sum(dim=-1).mean()

    return ratio.mean()

 
# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────

class WaveletFlowModule(pl.LightningModule):
    """
    Multi-scale wavelet flow matching module.

    Parameters
    ----------
    in_channels      : number of physical fields (C)
    field_names      : field names, used for logging and plotting
    data_shape       : (H, W) of the spatial domain
    n_steps_output   : number of predicted time steps (T)
    scale_weights    : per-scale loss weights; length must equal n_scales
    subband_weights  : per-subband loss weights; length must equal n_coeff
    wavelet_cfg      : WaveletConfig
    velocity_net_cfg : VelocityNetConfig — must include n_levels
    flow_matching_cfg: FlowMatchingConfig
    optimizer_cfg    : OptimizerConfig
    normalizer       : normalizer instance, e.g. The Well's ZScoreNormalization

    Decoupled-depth notes
    ---------------------
    n_scales  (from wavelet_cfg) — wavelet decomposition depth; controls
              injection points and the number of velocity outputs.
    n_levels  (from velocity_net_cfg) — UNet depth (down/upsamplings).
              Must satisfy n_levels >= n_scales.  Use the same n_levels for
              all configurations (wavelet and data-space) to keep the
              network capacity consistent.

    T-dimension fix
    ---------------
    x0 (the current frame, T=1) is tiled to (B, C, T, H, W) before being
    passed through _to_coeff_dict so that curr[i] and x_tau[i] always have
    the same channel count C*T*n_coeff_i.  The wavelet probe also uses T
    to produce the correct spatial shapes.
    """

    def __init__(
        self,
        in_channels      : int,
        field_names      : List[str],
        data_shape       : tuple,
        n_steps_output   : int,
        scale_weights    : List[float],
        subband_weights  : List[float] | None,
        wavelet_cfg      : WaveletConfig,
        velocity_net_cfg : VelocityNetConfig,
        flow_matching_cfg: FlowMatchingConfig,
        optimizer_cfg    : OptimizerConfig,
        normalizer       : Any,
    ):
        super().__init__()

        self.in_channels       = in_channels
        self.field_names       = field_names
        self.data_shape        = data_shape
        self.n_steps_output    = n_steps_output
        self.normalizer        = normalizer
        self.optimizer_cfg     = optimizer_cfg
        self.flow_matching_cfg = flow_matching_cfg

        # ── Wavelet decomposer ────────────────────────────────────────────────
        self.use_wavelet = wavelet_cfg.use_wavelet
        self.n_scales    = wavelet_cfg.n_scales if self.use_wavelet else 1

        self.wavelet_decomposer = (
            WaveletDecomposer2D(**wavelet_cfg.decomposer_kwargs())
            if self.use_wavelet else None
        )

        # Probe must run AFTER n_steps_output and n_scales are set.
        self.scale_shapes = self._probe_wavelet_shapes()

        assert len(scale_weights) == self.n_scales, (
            f"scale_weights has {len(scale_weights)} entries "
            f"but n_scales = {self.n_scales}"
        )
        self.register_buffer(
            'scale_weights',
            torch.tensor(scale_weights, dtype=torch.float32),
        )

        if not self.use_wavelet:
            # In data-space mode there is only one "subband" (the signal itself),
            # so subband_weights is forced to [1.0] regardless of config.
            sb_weights = [1.0]
        else:
            sb_weights = subband_weights

        if sb_weights is not None:
            self.register_buffer(
            'subband_weights',
            torch.tensor(sb_weights, dtype=torch.float32),
        )
        else:
            self.subband_weights = None
        

        # ── Network ───────────────────────────────────────────────────────────
        vn = velocity_net_cfg
        self.net = WaveletFlowUNet(
            scale_shapes     = self.scale_shapes,
            in_channels      = self.in_channels,
            n_steps_output   = self.n_steps_output,   # T-fix: needed for wav_ch
            n_scales         = self.n_scales,
            n_levels         = vn.n_levels,            # decoupled UNet depth
            context_channels = vn.context_in_channels,
            n_context_frames = vn.num_context_frames,
            cond_dim         = vn.cond_dim,
            num_scalar_conds = vn.num_cond,
            base_ch          = vn.base_ch,
            n_res            = vn.n_res,
        )

        self.experiment_manager = None

        self.save_hyperparameters({
            "in_channels":       in_channels,
            "field_names":       field_names,
            "data_shape":        data_shape,
            "n_steps_output":    n_steps_output,
            "scale_weights":     scale_weights,
            "subband_weights":   sb_weights,
            "wavelet_cfg":       wavelet_cfg.to_dict(),
            "velocity_net_cfg":  velocity_net_cfg.to_dict(),
            "flow_matching_cfg": flow_matching_cfg.to_dict(),
            "optimizer_cfg":     optimizer_cfg.to_dict(),
            "normalizer_stats": {
                "mean":       {k: v.tolist() for k, v in normalizer.means.items()},
                "std":        {k: v.tolist() for k, v in normalizer.stds.items()},
                "mean_delta": {k: v.tolist() for k, v in normalizer.means_delta.items()},
                "std_delta":  {k: v.tolist() for k, v in normalizer.stds_delta.items()},
            },
            "normalizer_core_field_names":          normalizer.core_field_names,
            "normalizer_core_constant_field_names": normalizer.core_constant_field_names,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Shape probe
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _probe_wavelet_shapes(self) -> Dict[int, dict]:
        """
        Run a dummy tensor through the wavelet decomposer to determine the
        spatial shape and coefficient count at each scale.
        """
        H, W  = self.data_shape
        dummy = torch.zeros(1, self.in_channels, self.n_steps_output, H, W)

        if self.use_wavelet:
            was = self.wavelet_decomposer.training
            self.wavelet_decomposer.eval()
            coeffs = self.wavelet_decomposer.forward_transform(dummy)
            self.wavelet_decomposer.train(was)
            return {
                i: {'H': c.shape[-2], 'W': c.shape[-1], 'n_coeff': c.shape[3]}
                for i, c in coeffs.items()
            }
        return {0: {'H': H, 'W': W, 'n_coeff': 1}}

    # ─────────────────────────────────────────────────────────────────────────
    # Wavelet helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _to_coeff_dict(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """(B, C, T, H, W) → {scale: (B, C·T·n_coeff, H_s, W_s)}"""
        if self.use_wavelet:
            raw = self.wavelet_decomposer.forward_transform(x)
            return {
                i: rearrange(v, 'B C T Coeff H W -> B (C T Coeff) H W')
                for i, v in raw.items()
            }
        return {0: rearrange(x, 'B C T H W -> B (C T) H W')}

    def _to_coeff_dict_x0(self, x0: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Convert the current frame x0 (B, C, 1, H, W) to coefficient dict,
        tiling to T = n_steps_output along the temporal axis first.
        """
        x0_tiled = x0.expand(-1, -1, self.n_steps_output, -1, -1)
        return self._to_coeff_dict(x0_tiled)

    def _from_coeff_dict(self, coeff: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        {scale: (B, C·T·n_coeff, H_s, W_s)} → (B, T, H, W, C)  denormalised.
        """
        if self.use_wavelet:
            raw = {
                i: rearrange(
                    v, 'B (C T Coeff) H W -> B C T Coeff H W',
                    C=self.in_channels,
                    T=self.n_steps_output,
                    Coeff=self.scale_shapes[i]['n_coeff'],
                )
                for i, v in coeff.items()
            }
            spatial = self.wavelet_decomposer.inverse_transform(raw)
        else:
            spatial = rearrange(
                coeff[0], 'B (C T) H W -> B C T H W',
                C=self.in_channels, T=self.n_steps_output,
            )
        return self.normalizer.denormalize_flattened(
            rearrange(spatial, 'B C T H W -> B T H W C'), mode='variable'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Batch preparation
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_batch(self, batch):
        """
        Normalise inputs, split into current frame x0, targets x1,
        past context xp, and scalar conditions.
        Returns (x0, x1, cond, B).
        x0 is (B, C, 1, H, W) — the most recent input frame.
        """
        device = self.device
        inp = self.normalizer.normalize_flattened(
            batch["input_fields"].to(device), mode='variable')
        out = self.normalizer.normalize_flattened(
            batch["output_fields"].to(device), mode='variable')

        x0 = rearrange(inp[:, -1:], 'B 1 H W C -> B C 1 H W')
        x1 = rearrange(out,         'B T H W C -> B C T H W')
        xp = rearrange(inp[:, :-1], 'B T H W C -> B C T H W')

        scalars = batch['constant_scalars'].to(device) \
                if 'constant_scalars' in batch.keys() else None

        cond = {
            'const'      : scalars,
            'past_states': xp,
        }
        return x0, x1, cond, x0.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch, log_prefix: str) -> torch.Tensor:
        """
        OT-CFM loss in wavelet space, weighted sum over scales.

        Uses _subband_vmse which normalises each wavelet subband (LL, LH,
        HL, HH) independently, preventing the model from suppressing
        low-energy detail coefficients.
        """
        x0, x1, cond, B = self._prepare_batch(batch)

        c0 = self._to_coeff_dict_x0(x0)
        c1 = self._to_coeff_dict(x1)

        t     = torch.rand(B, device=self.device).clamp(1e-5, 1.0 - 1e-5)
        t_exp = t.view(-1, 1, 1, 1)

        x_tau   = {i: t_exp * c1[i] + (1.0 - t_exp) * torch.randn_like(c1[i])
                   for i in range(self.n_scales)}
        targets = {i: c1[i] - (x_tau[i] - t_exp * c1[i]) / (1.0 - t_exp).clamp(min=1e-5)
                   for i in range(self.n_scales)}

        vel = self.net(
            curr        = c0,
            x_tau       = x_tau,
            t           = t,
            scalar_conds= cond['const'],
            context     = cond['past_states'],
        )

        w_sum = self.scale_weights.sum().clamp(min=1e-8)
        total = x0.new_zeros(())

        for i in range(self.n_scales):
            loss_i = _subband_vmse(
                pred            = vel[i],
                target          = targets[i],
                in_channels     = self.in_channels,
                n_steps_output  = self.n_steps_output,
                n_coeff         = self.scale_shapes[i]['n_coeff'],
                subband_weights = self.subband_weights,
            )
            total = total + (self.scale_weights[i] / w_sum) * loss_i
            self.log(f"{log_prefix}_loss_s{i}", loss_i,
                     on_epoch=True, on_step=(log_prefix == 'train'),
                     prog_bar=False, sync_dist=True, batch_size=B)

        self.log(f"{log_prefix}_loss", total,
                 on_epoch=True, on_step=(log_prefix == 'train'),
                 prog_bar=True, sync_dist=True, batch_size=B)
        
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Lightning steps
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'val')

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch, batch_idx, n_ensemble: int | None = None):
        """
        Euler integration of the learned vector field.
        """
        fm    = self.flow_matching_cfg
        n_ens = n_ensemble or fm.n_ensemble
        dt    = 1.0 / fm.num_sampling_steps

        x0, x1, cond, B = self._prepare_batch(batch)

        c0 = self._to_coeff_dict_x0(x0)
        c1 = self._to_coeff_dict(x1)

        members: List[Dict[int, torch.Tensor]] = []

        for _ in range(n_ens):
            x_tau = {i: torch.randn_like(c1[i]) for i in range(self.n_scales)}

            for step in range(fm.num_sampling_steps):
                t_batch = torch.full(
                    (B,), step / fm.num_sampling_steps,
                    device=self.device, dtype=x0.dtype,
                )
                vel = self.net(
                    curr        = c0,
                    x_tau       = x_tau,
                    t           = t_batch,
                    scalar_conds= cond['const'],
                    context     = cond['past_states'],
                )
                for i in range(self.n_scales):
                    x_tau[i] = x_tau[i] + dt * vel[i]

            members.append(self._from_coeff_dict(x_tau))

        yhat = torch.stack(members, dim=1)
        y = self.normalizer.denormalize_flattened(
            rearrange(x1, 'B C T H W -> B T H W C'), mode='variable'
        )
        return yhat, y

    # ─────────────────────────────────────────────────────────────────────────
    # Autoregressive rollout
    # ─────────────────────────────────────────────────────────────────────────

    def _to_spatial_normalized(self, coeff: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        {scale: (B, C·T·n_coeff, H_s, W_s)} → (B, T, H, W, C)  still NORMALIZED.
        """
        if self.use_wavelet:
            raw = {
                i: rearrange(
                    v, 'B (C T Coeff) H W -> B C T Coeff H W',
                    C=self.in_channels,
                    T=self.n_steps_output,
                    Coeff=self.scale_shapes[i]['n_coeff'],
                )
                for i, v in coeff.items()
            }
            spatial = self.wavelet_decomposer.inverse_transform(raw)
        else:
            spatial = rearrange(
                coeff[0], 'B (C T) H W -> B C T H W',
                C=self.in_channels, T=self.n_steps_output,
            )
        return rearrange(spatial, 'B C T H W -> B T H W C')

    @torch.no_grad()
    def rollout(
        self,
        seed            : torch.Tensor,
        scalar_conds    : torch.Tensor,
        rollout_steps   : int,
        n_steps_input   : int | None = None,
        n_ensemble      : int | None = None,
        normalized_input: bool = False,
        verbose         : bool = False,
        ode_method      : str  = "heun",
        clamp_sigma     : float = 5.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive rollout with configurable ODE solver.

        Returns
        -------
        preds_norm  : (B, E, rollout_steps, H, W, C)  normalized
        preds_phys  : (B, E, rollout_steps, H, W, C)  physical (denormalized)
        """
        fm     = self.flow_matching_cfg
        n_ens  = n_ensemble if n_ensemble is not None else fm.n_ensemble
        T_out  = self.n_steps_output
        device = self.device
        dtype  = next(self.parameters()).dtype

        if seed.ndim != 5:
            raise ValueError(f"seed must be (B, Ti, H, W, C), got {tuple(seed.shape)}")
        B, Ti_seed, H, W, C = seed.shape
        Ti = n_steps_input if n_steps_input is not None else Ti_seed
        if Ti_seed < Ti:
            raise ValueError(f"seed has {Ti_seed} frames but n_steps_input={Ti}.")

        seed = seed.to(device=device, dtype=dtype)

        if not normalized_input:
            seed_norm = self.normalizer.normalize_flattened(seed, mode='variable')
        else:
            seed_norm = seed
        seed_norm = seed_norm[:, -Ti:]

        windows = (
            seed_norm.unsqueeze(1)
            .expand(-1, n_ens, -1, -1, -1, -1)
            .reshape(B * n_ens, Ti, H, W, C)
            .clone()
        )

        if scalar_conds is not None:
            scalar_conds = scalar_conds.to(device=device, dtype=dtype)
            scalar_exp = (
                scalar_conds.unsqueeze(1)
                .expand(-1, n_ens, -1)
                .reshape(B * n_ens, -1)
            )
        else:
            scalar_exp = None

        BxE = B * n_ens
        preds_norm = torch.zeros(BxE, rollout_steps, H, W, C, device='cpu', dtype=dtype)

        step_out = 0
        n_calls  = math.ceil(rollout_steps / T_out)
        call_count = 0

        while step_out < rollout_steps:
            call_count += 1
            if verbose:
                print(
                    f"  rollout call {call_count:>4d}/{n_calls} "
                    f"(frames {step_out+1}–{min(step_out+T_out, rollout_steps)}"
                    f"/{rollout_steps})",
                    end='\r',
                )

            curr_norm = rearrange(windows[:, -1:], 'B 1 H W Cv -> B Cv 1 H W')
            past_norm = rearrange(windows[:, :-1], 'B T H W Cv -> B Cv T H W')
            c0 = self._to_coeff_dict_x0(curr_norm)

            x_tau = {
                i: torch.randn(
                    BxE,
                    self.in_channels * T_out * self.scale_shapes[i]['n_coeff'],
                    self.scale_shapes[i]['H'],
                    self.scale_shapes[i]['W'],
                    device=device, dtype=dtype,
                )
                for i in range(self.n_scales)
            }

            for s in range(fm.num_sampling_steps):
                t_s    = s / fm.num_sampling_steps
                t_next = (s + 1) / fm.num_sampling_steps
                dt_s   = t_next - t_s

                t_batch = torch.full((BxE,), t_s, device=device, dtype=dtype)

                if ode_method == "euler":
                    vel = self.net(
                        curr=c0, x_tau=x_tau, t=t_batch,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    for i in range(self.n_scales):
                        x_tau[i] = x_tau[i] + dt_s * vel[i]

                elif ode_method == "heun":
                    vel1 = self.net(
                        curr=c0, x_tau=x_tau, t=t_batch,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    x_tau_pred = {
                        i: x_tau[i] + dt_s * vel1[i]
                        for i in range(self.n_scales)
                    }
                    t_batch_next = torch.full((BxE,), t_next, device=device, dtype=dtype)
                    vel2 = self.net(
                        curr=c0, x_tau=x_tau_pred, t=t_batch_next,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    for i in range(self.n_scales):
                        x_tau[i] = x_tau[i] + dt_s * 0.5 * (vel1[i] + vel2[i])

                elif ode_method == "midpoint":
                    vel1 = self.net(
                        curr=c0, x_tau=x_tau, t=t_batch,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    x_tau_mid = {
                        i: x_tau[i] + 0.5 * dt_s * vel1[i]
                        for i in range(self.n_scales)
                    }
                    t_mid = torch.full((BxE,), t_s + 0.5 * dt_s, device=device, dtype=dtype)
                    vel_mid = self.net(
                        curr=c0, x_tau=x_tau_mid, t=t_mid,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    for i in range(self.n_scales):
                        x_tau[i] = x_tau[i] + dt_s * vel_mid[i]

                else:
                    raise ValueError(
                        f"Unknown ode_method={ode_method!r}. "
                        f"Choose 'euler', 'heun', or 'midpoint'."
                    )

            pred_norm = self._to_spatial_normalized(x_tau)

            frames_to_store = min(T_out, rollout_steps - step_out)
            preds_norm[:, step_out : step_out + frames_to_store] = (
                pred_norm[:, :frames_to_store].cpu()
            )
            step_out += frames_to_store

            if clamp_sigma > 0:
                pred_norm = pred_norm.clamp(-clamp_sigma, clamp_sigma)

            windows = torch.cat([windows, pred_norm], dim=1)[:, -Ti:]

        if verbose:
            print()

        preds_norm_out = preds_norm.view(B, n_ens, rollout_steps, H, W, C)
        preds_phys = self.normalizer.denormalize_flattened(
            preds_norm_out.reshape(B * n_ens * rollout_steps, H, W, C).to(device),
            mode='variable',
        ).reshape(B, n_ens, rollout_steps, H, W, C).cpu()

        return preds_norm_out, preds_phys

    # ─────────────────────────────────────────────────────────────────────────
    # Optimiser + scheduler
    # ─────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        oc  = self.optimizer_cfg
        opt = eval(oc.cls)(self.parameters(), **oc.optimizer_kwargs())

        if not oc.scheduler_cls:
            return opt

        cosine_epochs = max(self.trainer.max_epochs - oc.warmup_epochs, 1)
        sched = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                         total_iters=oc.warmup_epochs),
                CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=oc.eta_min),
            ],
            milestones=[oc.warmup_epochs],
        )
        return {
            "optimizer"   : opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch-end plotting
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_epoch_end(self, ensemble_reduce: str = 'random'):
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

        E = yhat.shape[1]
        if ensemble_reduce == 'mean':
            yhat_reduced = yhat.mean(dim=1)
        else:
            e = torch.randint(0, E, (1,)).item()
            yhat_reduced = yhat[:, e]

        self.experiment_manager.dump_sample_plots(
            yhat        = rearrange(yhat_reduced, 'B T H W C -> B C T H W'),
            y           = rearrange(y,            'B T H W C -> B C T H W'),
            field_names = self.field_names,
            fig_name    = f"epoch_{self.current_epoch}_val_batch_{batch_idx}.png",
            cmap        = 'RdBu_r',
            dpi         = 300,
        )

    def _get_random_validation_sample(self):
        loader = self.trainer.val_dataloaders
        idx    = torch.randint(0, len(loader), (1,)).item()
        for i, batch in enumerate(loader):
            if i == idx:
                break
        B = batch["output_fields"].shape[0]
        s = torch.randint(0, B, (1,)).item()
        return {k: v[s:s + 1] for k, v in batch.items()}, idx

    # ─────────────────────────────────────────────────────────────────────────
    # Portable checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu", stats: dict = None) -> "WaveletFlowModule":
        from the_well.data.normalization import ZScoreNormalization
        from waveletflux.utils.configs import (
            WaveletConfig, VelocityNetConfig, FlowMatchingConfig, OptimizerConfig,
        )

        ckpt = torch.load(path, map_location=map_location)
        hp   = ckpt["hyper_parameters"]

        if stats is not None:
            normalizer_stats = stats
        elif hp.get("normalizer_stats") is not None:
            normalizer_stats = hp["normalizer_stats"]
        elif "normalizer_means" in hp and "normalizer_stds" in hp:
            zeros = {k: 0.0 for k in hp["normalizer_means"]}
            normalizer_stats = {
                "mean":       hp["normalizer_means"],
                "std":        hp["normalizer_stds"],
                "mean_delta": zeros,
                "std_delta":  zeros,
            }
        else:
            raise KeyError(
                f"Cannot find normalizer stats in checkpoint. Keys: {list(hp.keys())}"
            )

        normalizer = ZScoreNormalization(
            stats                     = normalizer_stats,
            core_field_names          = hp["normalizer_core_field_names"],
            core_constant_field_names = hp["normalizer_core_constant_field_names"],
        )

        model = cls(
            in_channels       = hp["in_channels"],
            field_names       = hp["field_names"],
            data_shape        = tuple(hp["data_shape"]),
            n_steps_output    = hp["n_steps_output"],
            scale_weights     = hp["scale_weights"],
            subband_weights   = hp.get("subband_weights", None),
            wavelet_cfg       = WaveletConfig.from_dict(hp["wavelet_cfg"]),
            velocity_net_cfg  = VelocityNetConfig.from_dict(hp["velocity_net_cfg"]),
            flow_matching_cfg = FlowMatchingConfig.from_dict(hp["flow_matching_cfg"]),
            optimizer_cfg     = OptimizerConfig.from_dict(hp["optimizer_cfg"]),
            normalizer        = normalizer,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model

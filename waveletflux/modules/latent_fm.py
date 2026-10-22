r"""
latent_fm.py  —  Latent Flow Matching (LFM) baseline.

A drop-in sibling of :class:`WaveletFlowModule` that performs the *same* OT-CFM
flow-matching, with the *same* velocity U-Net, but in the latent space of a
**frozen, pre-trained LOLA autoencoder** instead of wavelet-coefficient space.

Design (mirrors the data-space / "Fourier FM" variant of WFM, i.e. n_scales=1)
-----------------------------------------------------------------------------
    physical fields  ──field_preprocess──►  encoder ─► z   (frozen AE, softclip2)
                                                   │
                             OT-CFM in latent space (single scale, n_coeff=1)
                                                   │
    physical fields  ◄─field_postprocess─◄  decoder ◄─ ẑ  (frozen AE)

Key parity with WaveletFlowModule
---------------------------------
  * identical FM interpolant / velocity target        (x1 - noise)
  * identical velocity network  (WaveletFlowUNet, n_scales=1)
  * identical conditioning: current frame injected at the stem, past frames via
    the ContextEncoder, scalar conditions (log Ra, log Pr) via FiLM
  * identical per-channel variance-normalised MSE loss (_subband_vmse, n_coeff=1)
  * identical ODE samplers (euler / heun / midpoint) and rollout API
  * identical public methods: training_step, validation_step, sample, rollout,
    configure_optimizers, on_train_epoch_end, from_checkpoint

Only necessary differences from WFM
-----------------------------------
  * pixel normalisation uses the AE's own field statistics (LOLA
    ``field_preprocess``), NOT The Well's ZScoreNormalization, so the encoder
    sees the distribution it was trained on;
  * the FM operates on the encoder's ``softclip2``-saturated latent (~[-5, 5]);
    no extra latent standardisation (matching LOLA);
  * ``n_levels`` (U-Net depth) must be chosen for the *latent* grid: a 512x128
    RB state becomes a 16x4 latent under the f32 autoencoders, so the pixel-space
    depth of 3 would collapse the width. Use n_levels=2 (see the config).

The autoencoder is frozen, excluded from the optimiser, and NOT written into the
checkpoint (it is rebuilt from its LOLA run at load time), keeping checkpoints
small and self-describing.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from typing import Any, Dict, List, Optional
import pytorch_lightning as pl
from einops import rearrange
from torch.optim import *  # noqa: F401,F403  (AdamW, etc. resolved by eval(cls))
import torch
import math

from waveletflux.models.wavelet_unet import WaveletFlowUNet
from waveletflux.modules.wavelet_fm import _subband_vmse
from waveletflux.utils.latent_norm import (
    field_preprocess,
    field_postprocess,
    load_lola_autoencoder,
)
from waveletflux.models.dcae import get_autoencoder
from waveletflux.utils.configs import (
    VelocityNetConfig,
    FlowMatchingConfig,
    OptimizerConfig,
)


class LatentFlowModule(pl.LightningModule):
    r"""Latent flow matching over a frozen LOLA autoencoder.

    Parameters
    ----------
    in_channels      : number of *physical* fields C (e.g. 4 for RB).
    field_names      : field names (logging / plotting).
    data_shape       : (H, W) of the *pixel* domain.
    n_steps_output   : number of predicted future frames T.
    velocity_net_cfg : VelocityNetConfig (n_levels here refers to the LATENT
                       U-Net depth; context_in_channels is ignored — the latent
                       channel count is used instead).
    flow_matching_cfg: FlowMatchingConfig.
    optimizer_cfg    : OptimizerConfig.
    autoencoder      : a frozen AutoEncoder (built by load_lola_autoencoder).
    ae_cfg           : the LOLA ``ae`` config dict (for checkpoint rebuild).
    ae_state_path    : path to the AE ``state.pth`` (for checkpoint rebuild).
    pix_mean, pix_std: per-field pixel statistics (LOLA field_preprocess).
    pix_transform    : per-field non-linear transform mapping ({} for RB).
    """

    # AE lives in the module but is not saved; tolerate its absence on resume.
    strict_loading = False

    def __init__(
        self,
        in_channels      : int,
        field_names      : List[str],
        data_shape       : tuple,
        n_steps_output   : int,
        velocity_net_cfg : VelocityNetConfig,
        flow_matching_cfg: FlowMatchingConfig,
        optimizer_cfg    : OptimizerConfig,
        autoencoder      : Any,
        ae_cfg           : Dict[str, Any],
        ae_state_path    : str,
        pix_mean         : torch.Tensor,
        pix_std          : torch.Tensor,
        pix_transform    : Optional[Dict[int, str]] = None,
    ):
        super().__init__()

        self.in_channels       = in_channels          # pixel field count C
        self.field_names       = field_names
        self.data_shape        = tuple(data_shape)
        self.n_steps_output    = n_steps_output
        self.T                 = n_steps_output
        self.optimizer_cfg     = optimizer_cfg
        self.flow_matching_cfg = flow_matching_cfg
        self.pix_transform     = dict(pix_transform or {})

        # ── Frozen autoencoder ────────────────────────────────────────────────
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad_(False)
        self._ae_cfg        = dict(ae_cfg)
        self._ae_state_path = str(ae_state_path)

        # Pixel-space field statistics (persistent, tiny)
        self.register_buffer("pix_mean", torch.as_tensor(pix_mean, dtype=torch.float32))
        self.register_buffer("pix_std",  torch.as_tensor(pix_std,  dtype=torch.float32))

        # ── Probe latent geometry from the AE ─────────────────────────────────
        self.lat_channels, self.lat_shape, self.spatial_factor = self._probe_latent()
        Hl, Wl = self.lat_shape

        # Single-scale "coefficient" description for the U-Net (n_coeff = 1).
        self.scale_shapes = {0: {"H": Hl, "W": Wl, "n_coeff": 1}}
        self.n_scales     = 1

        # ── Velocity network (same WaveletFlowUNet, data-space mode) ───────────
        vn = velocity_net_cfg
        n_context_frames = vn.num_context_frames
        self.net = WaveletFlowUNet(
            scale_shapes     = self.scale_shapes,
            in_channels      = self.lat_channels,      # ← latent channels, not C
            n_steps_output   = self.n_steps_output,
            n_scales         = 1,
            n_levels         = vn.n_levels,            # latent U-Net depth (see config)
            context_channels = self.lat_channels if n_context_frames > 0 else 0,
            n_context_frames = n_context_frames,
            cond_dim         = vn.cond_dim,
            num_scalar_conds = vn.num_cond,
            base_ch          = vn.base_ch,
            n_res            = vn.n_res,
        )

        self.experiment_manager = None

        self.save_hyperparameters({  # noqa: E128
            "in_channels":       in_channels,
            "field_names":       field_names,
            "data_shape":        list(self.data_shape),
            "n_steps_output":    n_steps_output,
            "velocity_net_cfg":  velocity_net_cfg.to_dict(),
            "flow_matching_cfg": flow_matching_cfg.to_dict(),
            "optimizer_cfg":     optimizer_cfg.to_dict(),
            "ae_cfg":            self._ae_cfg,
            "ae_state_path":     self._ae_state_path,
            "pix_mean":          self.pix_mean.tolist(),
            "pix_std":           self.pix_std.tolist(),
            "pix_transform":     {int(k): v for k, v in self.pix_transform.items()},
            "lat_channels":      self.lat_channels,
            "lat_shape":         list(self.lat_shape),
            "spatial_factor":    self.spatial_factor,
        })

    def train(self, mode: bool = True):
        """Keep the frozen autoencoder in eval mode (its ResBlocks use dropout)."""
        super().train(mode)
        self.autoencoder.eval()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Latent geometry probe
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _probe_latent(self):
        H, W = self.data_shape
        dummy = torch.zeros(1, self.in_channels, H, W)
        p = next(self.autoencoder.parameters(), None)
        if p is not None:
            dummy = dummy.to(p.device, p.dtype)
        z = self.autoencoder.encode(dummy)
        _, Clat, Hl, Wl = z.shape
        factor = H // Hl
        assert H % Hl == 0 and W % Wl == 0, (
            f"pixel shape {(H, W)} not divisible by latent shape {(Hl, Wl)}"
        )
        return Clat, (Hl, Wl), factor

    # ─────────────────────────────────────────────────────────────────────────
    # Encode / decode  (frozen AE, always fp32, autocast disabled)
    # ─────────────────────────────────────────────────────────────────────────

    def _encode(self, x_pix_norm: torch.Tensor) -> torch.Tensor:
        """(B, L, H, W, C) pixel-normalised  →  (B, Clat, L, Hl, Wl) latent."""
        B, L = x_pix_norm.shape[0], x_pix_norm.shape[1]
        x = rearrange(x_pix_norm, "B L H W C -> (B L) C H W").float()
        dev = x.device.type
        with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
            z = self.autoencoder.encode(x)
        return rearrange(z, "(B L) C H W -> B C L H W", B=B)

    def _decode(self, z_lat: torch.Tensor) -> torch.Tensor:
        """(B, L, Hl, Wl, Clat) latent  →  (B, L, H, W, C) pixel-normalised."""
        B, L = z_lat.shape[0], z_lat.shape[1]
        z = rearrange(z_lat, "B L Hl Wl C -> (B L) C Hl Wl").float()
        dev = z.device.type
        with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
            x = self.autoencoder.decode(z, noisy=False)
        return rearrange(x, "(B L) C H W -> B L H W C", B=B)

    def _decode_to_physical(self, z_lat: torch.Tensor) -> torch.Tensor:
        """(B, L, Hl, Wl, Clat) latent → (B, L, H, W, C) PHYSICAL fields."""
        x_norm = self._decode(z_lat)
        return field_postprocess(x_norm, self.pix_mean, self.pix_std,
                                 self.pix_transform, dim=-1)

    # latent <-> U-Net coefficient-dict bridges (single scale, n_coeff=1)
    def _lat_to_curr(self, x0_lat: torch.Tensor) -> Dict[int, torch.Tensor]:
        """(B, Clat, 1, Hl, Wl) → {0: (B, Clat*T, Hl, Wl)} (tiled to T)."""
        x0_tiled = x0_lat.expand(-1, -1, self.T, -1, -1)
        return {0: rearrange(x0_tiled, "B C T H W -> B (C T) H W")}

    def _lat_to_target(self, x1_lat: torch.Tensor) -> Dict[int, torch.Tensor]:
        """(B, Clat, T, Hl, Wl) → {0: (B, Clat*T, Hl, Wl)}."""
        return {0: rearrange(x1_lat, "B C T H W -> B (C T) H W")}

    def _coeff_to_lat_frames(self, c: Dict[int, torch.Tensor]) -> torch.Tensor:
        """{0: (B, Clat*T, Hl, Wl)} → (B, T, Hl, Wl, Clat)."""
        return rearrange(c[0], "B (C T) H W -> B T H W C",
                         C=self.lat_channels, T=self.T)

    # ─────────────────────────────────────────────────────────────────────────
    # Batch preparation
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_batch(self, batch):
        """
        Encode raw physical fields into latent space.

        Returns (x0, x1, cond, B) with
          x0   : (B, Clat, 1, Hl, Wl)   current latent frame
          x1   : (B, Clat, T, Hl, Wl)   target latent frame(s)
          cond : {'const': scalars, 'past_states': (B, Clat, Ti-1, Hl, Wl)}
        """
        device = self.device
        inp = batch["input_fields"].to(device)    # (B, Ti, H, W, C) raw physical
        out = batch["output_fields"].to(device)   # (B, T,  H, W, C) raw physical

        inp_n = field_preprocess(inp, self.pix_mean, self.pix_std, self.pix_transform, dim=-1)
        out_n = field_preprocess(out, self.pix_mean, self.pix_std, self.pix_transform, dim=-1)

        lat_inp = self._encode(inp_n)   # (B, Clat, Ti, Hl, Wl)
        lat_out = self._encode(out_n)   # (B, Clat, T,  Hl, Wl)

        x0 = lat_inp[:, :, -1:]         # current latent frame
        xp = lat_inp[:, :, :-1]         # past latent context
        x1 = lat_out

        scalars = batch["constant_scalars"].to(device) \
            if "constant_scalars" in batch.keys() else None

        cond = {"const": scalars, "past_states": xp}
        return x0, x1, cond, x0.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Loss (OT-CFM in latent space)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch, log_prefix: str) -> torch.Tensor:
        x0, x1, cond, B = self._prepare_batch(batch)

        c0 = self._lat_to_curr(x0)      # conditioning (current frame)
        c1 = self._lat_to_target(x1)    # FM endpoint (target frame)

        t     = torch.rand(B, device=self.device).clamp(1e-5, 1.0 - 1e-5)
        t_exp = t.view(-1, 1, 1, 1)

        noise  = torch.randn_like(c1[0])
        x_tau  = {0: t_exp * c1[0] + (1.0 - t_exp) * noise}
        target = {0: c1[0] - noise}     # OT-CFM velocity  (== WFM's target)

        vel = self.net(
            curr         = c0,
            x_tau        = x_tau,
            t            = t,
            scalar_conds = cond["const"],
            context      = cond["past_states"],
        )

        loss = _subband_vmse(
            pred            = vel[0],
            target          = target[0],
            in_channels     = self.lat_channels,
            n_steps_output  = self.n_steps_output,
            n_coeff         = 1,
            subband_weights = None,
        )

        self.log(f"{log_prefix}_loss", loss,
                 on_epoch=True, on_step=(log_prefix == "train"),
                 prog_bar=True, sync_dist=True, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    # ─────────────────────────────────────────────────────────────────────────
    # ODE integration of one latent transition
    # ─────────────────────────────────────────────────────────────────────────

    def _integrate(self, c0, past, scalar_conds, ode_method):
        """Integrate the learned velocity field from noise to a latent frame.

        Returns the coefficient dict {0: (B, Clat*T, Hl, Wl)} at flow time 1.
        """
        fm  = self.flow_matching_cfg
        BxE = c0[0].shape[0]
        dev, dt = self.device, next(self.net.parameters()).dtype

        x_tau = {0: torch.randn(
            BxE, self.lat_channels * self.T, self.lat_shape[0], self.lat_shape[1],
            device=dev, dtype=dt,
        )}

        for s in range(fm.num_sampling_steps):
            t_s    = s / fm.num_sampling_steps
            t_next = (s + 1) / fm.num_sampling_steps
            dt_s   = t_next - t_s
            t_batch = torch.full((BxE,), t_s, device=dev, dtype=dt)

            if ode_method == "euler":
                vel = self.net(curr=c0, x_tau=x_tau, t=t_batch,
                               scalar_conds=scalar_conds, context=past)
                x_tau[0] = x_tau[0] + dt_s * vel[0]

            elif ode_method == "heun":
                vel1 = self.net(curr=c0, x_tau=x_tau, t=t_batch,
                                scalar_conds=scalar_conds, context=past)
                x_pred = {0: x_tau[0] + dt_s * vel1[0]}
                t_next_b = torch.full((BxE,), t_next, device=dev, dtype=dt)
                vel2 = self.net(curr=c0, x_tau=x_pred, t=t_next_b,
                                scalar_conds=scalar_conds, context=past)
                x_tau[0] = x_tau[0] + dt_s * 0.5 * (vel1[0] + vel2[0])

            elif ode_method == "midpoint":
                vel1 = self.net(curr=c0, x_tau=x_tau, t=t_batch,
                                scalar_conds=scalar_conds, context=past)
                x_mid = {0: x_tau[0] + 0.5 * dt_s * vel1[0]}
                t_mid = torch.full((BxE,), t_s + 0.5 * dt_s, device=dev, dtype=dt)
                vel_mid = self.net(curr=c0, x_tau=x_mid, t=t_mid,
                                   scalar_conds=scalar_conds, context=past)
                x_tau[0] = x_tau[0] + dt_s * vel_mid[0]

            else:
                raise ValueError(
                    f"Unknown ode_method={ode_method!r}. Choose 'euler', 'heun', or 'midpoint'."
                )

        return x_tau

    # ─────────────────────────────────────────────────────────────────────────
    # One-step sampling (validation plots)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch, batch_idx, n_ensemble: int | None = None,
               ode_method: str = "euler"):
        fm    = self.flow_matching_cfg
        n_ens = n_ensemble or fm.n_ensemble

        x0, x1, cond, B = self._prepare_batch(batch)
        c0 = self._lat_to_curr(x0)

        members = []
        for _ in range(n_ens):
            x_tau = self._integrate(c0, cond["past_states"], cond["const"], ode_method)
            lat_frames = self._coeff_to_lat_frames(x_tau)        # (B, T, Hl, Wl, Clat)
            members.append(self._decode_to_physical(lat_frames)) # (B, T, H, W, C)

        yhat = torch.stack(members, dim=1)                       # (B, E, T, H, W, C)

        # Ground truth, decoded through the AE for a fair (reconstruction-bounded) target
        y_lat = rearrange(x1, "B C T H W -> B T H W C")
        y     = self._decode_to_physical(y_lat)                  # (B, T, H, W, C)
        return yhat, y

    # ─────────────────────────────────────────────────────────────────────────
    # Autoregressive rollout  (identical API to WaveletFlowModule.rollout)
    # ─────────────────────────────────────────────────────────────────────────

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
        Autoregressive latent rollout.

        Returns
        -------
        preds_norm : (B, E, rollout_steps, H, W, C)  decoded, pixel-normalised
        preds_phys : (B, E, rollout_steps, H, W, C)  physical (denormalised)
        """
        fm     = self.flow_matching_cfg
        n_ens  = n_ensemble if n_ensemble is not None else fm.n_ensemble
        T_out  = self.n_steps_output
        device = self.device
        dtype  = next(self.net.parameters()).dtype

        if seed.ndim != 5:
            raise ValueError(f"seed must be (B, Ti, H, W, C), got {tuple(seed.shape)}")
        B, Ti_seed, H, W, C = seed.shape
        Ti = n_steps_input if n_steps_input is not None else Ti_seed
        if Ti_seed < Ti:
            raise ValueError(f"seed has {Ti_seed} frames but n_steps_input={Ti}.")

        seed = seed.to(device=device)

        # Pixel normalisation the AE expects (unless caller already did it).
        if not normalized_input:
            seed_pix = field_preprocess(seed.float(), self.pix_mean, self.pix_std,
                                        self.pix_transform, dim=-1)
        else:
            seed_pix = seed.float()
        seed_pix = seed_pix[:, -Ti:]                             # (B, Ti, H, W, C)

        # Encode the seed window into latent frames, expand to the ensemble.
        seed_lat = self._encode(seed_pix)                        # (B, Clat, Ti, Hl, Wl)
        seed_lat = rearrange(seed_lat, "B C T H W -> B T H W C") # (B, Ti, Hl, Wl, Clat)
        Hl, Wl = self.lat_shape

        windows = (
            seed_lat.unsqueeze(1)
            .expand(-1, n_ens, -1, -1, -1, -1)
            .reshape(B * n_ens, Ti, Hl, Wl, self.lat_channels)
            .clone()
            .to(dtype)
        )

        if scalar_conds is not None:
            scalar_conds = scalar_conds.to(device=device, dtype=dtype)
            scalar_exp = (
                scalar_conds.unsqueeze(1).expand(-1, n_ens, -1).reshape(B * n_ens, -1)
            )
        else:
            scalar_exp = None

        BxE = B * n_ens
        preds_lat = torch.zeros(BxE, rollout_steps, Hl, Wl, self.lat_channels,
                                device="cpu", dtype=dtype)

        step_out   = 0
        n_calls    = math.ceil(rollout_steps / T_out)
        call_count = 0

        while step_out < rollout_steps:
            call_count += 1
            if verbose:
                print(f"  rollout call {call_count:>4d}/{n_calls} "
                      f"(frames {step_out + 1}-{min(step_out + T_out, rollout_steps)}"
                      f"/{rollout_steps})", end="\r")

            curr_lat = rearrange(windows[:, -1:], "B 1 H W C -> B C 1 H W")
            past_lat = rearrange(windows[:, :-1], "B T H W C -> B C T H W")
            c0 = self._lat_to_curr(curr_lat)

            x_tau = self._integrate(c0, past_lat, scalar_exp, ode_method)
            pred_lat = self._coeff_to_lat_frames(x_tau)          # (BxE, T, Hl, Wl, Clat)

            frames_to_store = min(T_out, rollout_steps - step_out)
            preds_lat[:, step_out:step_out + frames_to_store] = (
                pred_lat[:, :frames_to_store].to("cpu", dtype)
            )
            step_out += frames_to_store

            if clamp_sigma > 0:
                pred_lat = pred_lat.clamp(-clamp_sigma, clamp_sigma)

            windows = torch.cat([windows, pred_lat.to(windows)], dim=1)[:, -Ti:]

        if verbose:
            print()

        # Decode all predicted latent frames to pixel space (chunked over frames).
        preds_lat = preds_lat.view(B, n_ens, rollout_steps, Hl, Wl, self.lat_channels)
        flat = preds_lat.reshape(BxE, rollout_steps, Hl, Wl, self.lat_channels)

        pix_norm = self._decode(flat.to(device))                          # (BxE, R, H, W, C)
        pix_phys = field_postprocess(pix_norm, self.pix_mean, self.pix_std,
                                     self.pix_transform, dim=-1)

        preds_norm = pix_norm.reshape(B, n_ens, rollout_steps, H, W, C).cpu()
        preds_phys = pix_phys.reshape(B, n_ens, rollout_steps, H, W, C).cpu()
        return preds_norm, preds_phys

    # ─────────────────────────────────────────────────────────────────────────
    # Optimiser + scheduler  (only trainable params; AE is frozen)
    # ─────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        oc = self.optimizer_cfg
        params = [p for p in self.parameters() if p.requires_grad]
        opt = eval(oc.cls)(params, **oc.optimizer_kwargs())

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
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch-end plotting  (mirrors WaveletFlowModule)
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_epoch_end(self, ensemble_reduce: str = "random"):
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
        if ensemble_reduce == "mean":
            yhat_reduced = yhat.mean(dim=1)
        else:
            e = torch.randint(0, E, (1,)).item()
            yhat_reduced = yhat[:, e]

        self.experiment_manager.dump_sample_plots(
            yhat        = rearrange(yhat_reduced, "B T H W C -> B C T H W"),
            y           = rearrange(y,            "B T H W C -> B C T H W"),
            field_names = self.field_names,
            fig_name    = f"epoch_{self.current_epoch}_val_batch_{batch_idx}.png",
            cmap        = "RdBu_r",
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
    # Checkpoint plumbing — keep the frozen AE out of the checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if k.startswith("autoencoder.")]:
            del sd[k]

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("autoencoder.")}

    # ─────────────────────────────────────────────────────────────────────────
    # Portable checkpoint  (rebuilds the frozen AE from its LOLA run)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        map_location="cpu",
        ae_state_path: Optional[str] = None,
    ) -> "LatentFlowModule":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hp   = ckpt["hyper_parameters"]

        # Rebuild the frozen autoencoder from its stored config + weights.
        ae_cfg    = dict(hp["ae_cfg"])
        state_pth = ae_state_path or hp["ae_state_path"]
        autoencoder = get_autoencoder(**ae_cfg)
        ae_state = torch.load(state_pth, map_location=map_location, weights_only=True)
        missing, unexpected = autoencoder.load_state_dict(ae_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"autoencoder weight mismatch loading {state_pth} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad_(False)

        model = cls(
            in_channels       = hp["in_channels"],
            field_names       = hp["field_names"],
            data_shape        = tuple(hp["data_shape"]),
            n_steps_output    = hp["n_steps_output"],
            velocity_net_cfg  = VelocityNetConfig.from_dict(hp["velocity_net_cfg"]),
            flow_matching_cfg = FlowMatchingConfig.from_dict(hp["flow_matching_cfg"]),
            optimizer_cfg     = OptimizerConfig.from_dict(hp["optimizer_cfg"]),
            autoencoder       = autoencoder,
            ae_cfg            = ae_cfg,
            ae_state_path     = state_pth,
            pix_mean          = torch.as_tensor(hp["pix_mean"]),
            pix_std           = torch.as_tensor(hp["pix_std"]),
            pix_transform     = {int(k): v for k, v in hp.get("pix_transform", {}).items()},
        )
        # Load everything except the AE (already loaded above).
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model

"""
latent_fm.py — Flow Matching in the VAE Latent Space.

Structural analogue of ``wavelet_fm.py``.  The ONLY architectural difference
is that the multi-scale wavelet decomposer is replaced by a frozen
AutoencoderKL, and _to_latent / _from_latent replace _to_coeff_dict /
_from_coeff_dict.  Everything else — _prepare_batch, OT-CFM formulation,
loss, configure_optimizers, sample, on_train_epoch_end — is a direct
copy/adaptation from wavelet_fm.py with no extra logic.

Correspondence table
────────────────────
wavelet_fm                  latent_fm
─────────────────────────   ─────────────────────────────────
WaveletDecomposer2D         AutoencoderKL (frozen)
_probe_wavelet_shapes       _probe_latent_shape
_to_coeff_dict              _to_latent
_to_coeff_dict_x0           _to_latent  (with expand, see _compute_loss)
_from_coeff_dict            _from_latent
WaveletConfig               LatentVAEConfig
n_scales = wavelet depth    n_scales = 1  (always, no hierarchy)

Channel grouping
────────────────
The SD-VAE encodes exactly 3-channel RGB tensors.  Physical fields have C
channels.  We group them into G = ceil(C/3) triples, padding the last group
with zeros when C % 3 != 0, and process them as a batch of G images.

  (B, C, H, W) → pad → (B, G*3, H, W) → (B*G, 3, H, W)
              → VAE → (B*G, 4, H_lat, W_lat)
              → reassemble → (B, G*4, H_lat, W_lat)

Per-sample rescaling
────────────────────
Before VAE encoding each (B*G) sub-image is rescaled so its absolute max
is at most 1.0.  The scale tensor is stored alongside the latent and
reapplied on decode.  At inference the scale of the predicted latent is
unknown — we fall back to scale=1.0.

TOML changes vs wavelet config
───────────────────────────────
Replace [wavelet] with [vae].
scale_weights must have exactly 1 entry (single latent scale).
context_in_channels must equal C (the number of physical fields) — the
same rule as wavelet_fm, because past_states context is passed in
physical space to ContextEncoder, never encoded to the latent space.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from typing import Any, Dict, List, Optional, Tuple
from diffusers import AutoencoderKL
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from torch.optim import *
import torch
import math

from waveletflux.models.wavelet_unet import WaveletFlowUNet
from waveletflux.utils.configs import (
    VelocityNetConfig,
    FlowMatchingConfig,
    OptimizerConfig,
)
from waveletflux.utils.configs import LatentVAEConfig


# ─────────────────────────────────────────────────────────────────────────────
# Loss  (identical to wavelet_fm._normed_mse)
# ─────────────────────────────────────────────────────────────────────────────

def _normed_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Variance-normalised MSE.
    var shape (B, C, 1, 1): per-channel spatial variance.
    Normalises across latent channels with very different energy levels.
    """
    var = target.var(dim=(2, 3), keepdim=True).detach().clamp(min=1e-4)
    return F.mse_loss(pred / var.sqrt(), target / var.sqrt())


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────

class LatentFlowModule(pl.LightningModule):
    """
    Flow matching in the VAE latent space.

    Parameters mirror WaveletFlowModule exactly.
    Replace wavelet_cfg with vae_cfg and set scale_weights = [1.0].
    """

    def __init__(
        self,
        in_channels:       int,
        field_names:       List[str],
        data_shape:        Tuple[int, int],
        n_steps_output:    int,
        scale_weights:     List[float],          # must be [1.0]
        vae_cfg:           LatentVAEConfig,
        velocity_net_cfg:  VelocityNetConfig,
        flow_matching_cfg: FlowMatchingConfig,
        optimizer_cfg:     OptimizerConfig,
        normalizer:        Any,
    ):
        super().__init__()

        self.in_channels       = in_channels
        self.field_names       = field_names
        self.data_shape        = data_shape
        self.n_steps_output    = n_steps_output
        self.normalizer        = normalizer
        self.optimizer_cfg     = optimizer_cfg
        self.flow_matching_cfg = flow_matching_cfg
        self.vae_cfg           = vae_cfg

        # ── Frozen VAE ────────────────────────────────────────────────────────
        #self.vae = AutoencoderKL.from_pretrained(vae_cfg.model_id, subfolder='vae')
        self.vae = AutoencoderKL.from_pretrained(vae_cfg.model_id)
        if vae_cfg.freeze:
            self.vae.requires_grad_(False)
            self.vae.eval()

        # Channel grouping: SD-VAE expects 3-channel RGB inputs
        self.n_groups     = math.ceil(in_channels / 3)
        self.pad_channels = self.n_groups * 3 - in_channels

        # Always a single latent scale — no wavelet hierarchy
        self.n_scales = 1

        # Probe latent shape — mirrors _probe_wavelet_shapes
        self.scale_shapes = self._probe_latent_shape()

        # latent_channels per frame = G * lat_ch_per_group
        # n_coeff in scale_shapes[0] already includes T, so divide it back out
        #self.latent_channels = (
        #    self.scale_shapes[0]['n_coeff'] // self.n_steps_output
        #)
        self.latent_channels = self.n_groups * self._lat_ch_per_group

        assert len(scale_weights) == 1, (
            f"LatentFlowModule uses a single latent scale; "
            f"scale_weights must have exactly 1 entry (got {len(scale_weights)})."
        )
        self.register_buffer(
            'scale_weights',
            torch.tensor(scale_weights, dtype=torch.float32),
        )

        # ── Velocity network — identical call to wavelet_fm ───────────────────
        # in_channels is set to latent_channels (G*4) because the UNet operates
        # in the latent domain.  context_in_channels stays equal to the number
        # of physical fields C because past_states context is passed in physical
        # space directly (never VAE-encoded), exactly as in wavelet_fm.
        vn = velocity_net_cfg
        self.net = WaveletFlowUNet(
            scale_shapes     = self.scale_shapes,
            in_channels      = self.latent_channels,
            n_steps_output   = n_steps_output,
            n_scales         = 1,
            n_levels         = vn.n_levels,
            context_channels = vn.context_in_channels,  # must equal physical C
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
            "data_shape":        list(data_shape),
            "n_steps_output":    n_steps_output,
            "scale_weights":     scale_weights,
            "vae_cfg":           vae_cfg.to_dict(),
            "velocity_net_cfg":  vn.to_dict(),
            "flow_matching_cfg": flow_matching_cfg.to_dict(),
            "optimizer_cfg":     optimizer_cfg.to_dict(),
            "normalizer_means":                     normalizer.means,
            "normalizer_stds":                      normalizer.stds,
            "normalizer_core_field_names":          normalizer.core_field_names,
            "normalizer_core_constant_field_names": normalizer.core_constant_field_names,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Shape probe  (mirrors _probe_wavelet_shapes)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _probe_latent_shape(self) -> Dict[int, dict]:
        H, W   = self.data_shape
        device = next(self.vae.parameters()).device
        dummy  = torch.zeros(self.n_groups, 3, H, W, device=device)
        lat    = self.vae.encode(dummy).latent_dist.mode()
        _, lat_ch_per_group, H_lat, W_lat = lat.shape

        # Save so __init__ can compute latent_channels directly
        self._lat_ch_per_group = lat_ch_per_group

        # n_coeff = 1: there is only one "sub-band" in latent space.
        # G and lat_ch are already in `in_channels = latent_channels`.
        # T is already in wav_ch via n_steps_output. No double-counting.
        return {0: {'H': H_lat, 'W': W_lat, 'n_coeff': 1}}

    #@torch.no_grad()
    #def _probe_latent_shape(self) -> Dict[int, dict]:
    #    """
    #    Run a dummy tensor through the VAE encoder to determine latent dims.
#
    #    Returns {0: {'H': H_lat, 'W': W_lat, 'n_coeff': G*lat_ch*T}} so
    #    that WaveletFlowUNet receives the same scale_shapes structure as
    #    in wavelet_fm, and wav_ch[0] = latent_channels * T correctly.
    #    """
    #    H, W   = self.data_shape
    #    device = next(self.vae.parameters()).device
    #    dummy  = torch.zeros(self.n_groups, 3, H, W, device=device)
    #    lat    = self.vae.encode(dummy).latent_dist.mode()
    #    _, lat_ch_per_group, H_lat, W_lat = lat.shape
#
    #    # n_coeff = G * lat_ch_per_group * T  →  wav_ch[0] = latent_ch * T
    #    n_coeff = self.n_groups * lat_ch_per_group * self.n_steps_output
#
    #    return {0: {'H': H_lat, 'W': W_lat, 'n_coeff': n_coeff}}

    # ─────────────────────────────────────────────────────────────────────────
    # VAE encode / decode  (internal building blocks)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_frame(
        self,
        x_bchw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one (B, C, H, W) normalised frame → (B, G*lat_ch, H_lat, W_lat).
        Also returns per-group scale (B*G, 1, 1, 1) for faithful decode.
        """
        B      = x_bchw.shape[0]
        x_pad  = F.pad(x_bchw, (0, 0, 0, 0, 0, self.pad_channels))
        packed = rearrange(x_pad, 'b (g rgb) h w -> (b g) rgb h w', rgb=3)
        scale  = packed.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        post   = self.vae.encode(packed / scale).latent_dist
        lat_bg = (post.mode() if self.vae_cfg.use_mode else post.sample())
        lat_bg = lat_bg * self.vae.config.scaling_factor * self.vae_cfg.latent_scale
        latent = rearrange(lat_bg, '(b g) c h w -> b (g c) h w', b=B, g=self.n_groups)
        return latent, scale

    @torch.no_grad()
    def _decode_frame(
        self,
        latent: torch.Tensor,
        scale:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode (B, G*lat_ch, H_lat, W_lat) → (B, C, H, W).
        scale=None → use 1.0 (inference fallback when x1 scale is unknown).
        """
        B      = latent.shape[0]
        if self.vae_cfg.latent_scale != 1.0:
            latent = latent / self.vae_cfg.latent_scale
        lat_bg = rearrange(latent, 'b (g c) h w -> (b g) c h w', g=self.n_groups)
        recon  = self.vae.decode(lat_bg / self.vae.config.scaling_factor).sample
        if scale is not None:
            recon = recon * scale
        x_pad  = rearrange(recon, '(b g) rgb h w -> b (g rgb) h w',
                           b=B, g=self.n_groups)
        return x_pad[:, :self.in_channels]

    # ─────────────────────────────────────────────────────────────────────────
    # Latent dict helpers  (mirror _to_coeff_dict / _from_coeff_dict)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _to_latent(
        self,
        x_bcthw: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        (B, C, T, H, W)  →  ({0: (B, G*lat_ch*T, H_lat, W_lat)}, scales)

        Mirrors _to_coeff_dict: packs all T frames into the channel axis so
        WaveletFlowUNet always receives a tensor with wav_ch[0] = lat_ch*T.
        scales is (T*B*G, 1, 1, 1) — one per (frame, sample, group).
        """
        B, C, T = x_bcthw.shape[:3]
        lat_list, scale_list = [], []
        for t in range(T):
            z_t, s_t = self._encode_frame(x_bcthw[:, :, t])
            lat_list.append(z_t)
            scale_list.append(s_t)
        return {0: torch.cat(lat_list, dim=1)}, torch.cat(scale_list, dim=0)

    def _from_latent(
        self,
        latent_dict: Dict[int, torch.Tensor],
        scales:      Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        {0: (B, G*lat_ch*T, H_lat, W_lat)}  →  (B, T, H, W, C) denormalised.

        Mirrors _from_coeff_dict.
        """
        lat_all = latent_dict[0]
        B       = lat_all.shape[0]
        lat_ch  = self.latent_channels
        T       = self.n_steps_output

        frames = []
        for t in range(T):
            z_t  = lat_all[:, t * lat_ch : (t + 1) * lat_ch]
            #sc_t = scales[t * B : (t + 1) * B] if scales is not None else None
            sc_t = scales[t * B * self.n_groups : (t + 1) * B * self.n_groups] if scales is not None else None
            frames.append(self._decode_frame(z_t, sc_t))

        spatial = torch.stack(frames, dim=2)   # (B, C, T, H, W)
        return self.normalizer.denormalize_flattened(
            rearrange(spatial, 'B C T H W -> B T H W C'), mode='variable'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Batch preparation  (identical to wavelet_fm._prepare_batch)
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_batch(self, batch):
        """
        Identical to wavelet_fm._prepare_batch.

        Returns (x0, x1, cond, B).
          x0                  : (B, C, 1, H, W)    most recent input frame
          x1                  : (B, C, T, H, W)    target frames
          cond['const']       : (B, num_cond)       scalar physical conditions
          cond['past_states'] : (B, C, T_past, H, W) past context in physical space
        """
        device = self.device
        inp = self.normalizer.normalize_flattened(
            batch["input_fields"].to(device), mode='variable')
        out = self.normalizer.normalize_flattened(
            batch["output_fields"].to(device), mode='variable')

        x0 = rearrange(inp[:, -1:], 'B 1 H W C -> B C 1 H W')
        x1 = rearrange(out,         'B T H W C -> B C T H W')
        xp = rearrange(inp[:, :-1], 'B T H W C -> B C T H W')

        cond = {
            'const'      : batch['constant_scalars'].to(device),  # (B, num_cond)
            'past_states': xp,                                     # (B, C, T_past, H, W)
        }
        return x0, x1, cond, x0.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Loss  (mirrors wavelet_fm._compute_loss)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch, log_prefix: str) -> torch.Tensor:
        """
        OT-CFM loss in the VAE latent space.

        Identical formulation to wavelet_fm._compute_loss:
          c0 = encode(x0 tiled to T)
          c1 = encode(x1)
          ε  ~ N(0, I)
          x_tau[0]  = t·c1[0] + (1−t)·ε        OT-CFM interpolant
          targets[0]= c1[0] − ε                 conditional velocity
          vel       = net(c0, x_tau, t, scalar_conds, context)
          loss      = normed_mse(vel[0], targets[0])
        """
        x0, x1, cond, B = self._prepare_batch(batch)

        # Tile x0 to T frames (mirrors _to_coeff_dict_x0)
        x0_tiled   = x0.expand(-1, -1, self.n_steps_output, -1, -1)
        c0, _      = self._to_latent(x0_tiled)
        c1, _      = self._to_latent(x1)

        t     = torch.rand(B, device=self.device).clamp(1e-5, 1.0 - 1e-5)
        t_exp = t.view(-1, 1, 1, 1)

        x_tau   = {0: t_exp * c1[0] + (1.0 - t_exp) * torch.randn_like(c1[0])}
        targets = {0: c1[0] - (x_tau[0] - t_exp * c1[0]) / (1.0 - t_exp).clamp(min=1e-5)}

        vel = self.net(
            curr         = c0,
            x_tau        = x_tau,
            t            = t,
            scalar_conds = cond['const'],
            context      = cond['past_states'],
        )

        w_sum  = self.scale_weights.sum().clamp(min=1e-8)
        loss_0 = _normed_mse(vel[0], targets[0])
        total  = (self.scale_weights[0] / w_sum) * loss_0

        self.log(f"{log_prefix}_loss_s0", loss_0,
                 on_epoch=True, on_step=(log_prefix == 'train'),
                 prog_bar=False, sync_dist=True, batch_size=B)
        self.log(f"{log_prefix}_loss", total,
                 on_epoch=True, on_step=(log_prefix == 'train'),
                 prog_bar=True, sync_dist=True, batch_size=B)
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Lightning steps  (identical to wavelet_fm)
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'val')

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling  (mirrors wavelet_fm.sample)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch, batch_idx, n_ensemble: int | None = None):
        """
        Euler integration in the VAE latent space.

        Identical structure to wavelet_fm.sample; replace wavelet coefficients
        with latent tensors and IDWT with VAE decode.
        """
        fm    = self.flow_matching_cfg
        n_ens = n_ensemble or fm.n_ensemble
        dt    = 1.0 / fm.num_sampling_steps

        x0, x1, cond, B = self._prepare_batch(batch)

        x0_tiled   = x0.expand(-1, -1, self.n_steps_output, -1, -1)
        c0, _      = self._to_latent(x0_tiled)
        c1, scales = self._to_latent(x1)
        #c1, _ = self._to_latent(x1)

        members: List[Dict[int, torch.Tensor]] = []

        for _ in range(n_ens):
            x_tau = {0: torch.randn_like(c1[0])}

            for step in range(fm.num_sampling_steps):
                t_batch = torch.full(
                    (B,), step / fm.num_sampling_steps,
                    device=self.device, dtype=x0.dtype,
                )
                vel = self.net(
                    curr         = c0,
                    x_tau        = x_tau,
                    t            = t_batch,
                    scalar_conds = cond['const'],
                    context      = cond['past_states'],
                )
                x_tau[0] = x_tau[0] + dt * vel[0]

            members.append(self._from_latent(x_tau, scales))
            #members.append(self._from_latent(x_tau, None))

        yhat = torch.stack(members, dim=1)      # (B, E, T, H, W, C)
        #y    = self._from_latent(c1, scales)   # (B, T, H, W, C)
        y = self.normalizer.denormalize_flattened(rearrange(x1, 'B C T H W -> B T H W C'), mode='variable')
        return yhat, y

    # ─────────────────────────────────────────────────────────────────────────
    # Optimiser + scheduler  (identical to wavelet_fm.configure_optimizers)
    # ─────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        oc  = self.optimizer_cfg
        # Only the velocity UNet trains; the VAE is frozen
        opt = eval(oc.cls)(
            [p for p in self.net.parameters() if p.requires_grad],
            **oc.optimizer_kwargs(),
        )

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
    # Epoch-end plotting  (mirrors wavelet_fm.on_train_epoch_end)
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
    # Portable checkpoint  (mirrors wavelet_fm.from_checkpoint)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "LatentFlowModule":
        """
        Load a fully self-contained checkpoint.
        No TOML config, no distributed setup — just the .ckpt file.
        """
        from the_well.data.normalization import ZScoreNormalization
        from waveletflux.utils.configs import (
            VelocityNetConfig, FlowMatchingConfig, OptimizerConfig,
        )

        ckpt = torch.load(path, map_location=map_location)
        hp   = ckpt["hyper_parameters"]

        normalizer = ZScoreNormalization(
            stats={
                'mean': hp["normalizer_means"],
                'std' : hp["normalizer_stds"],
            },
            core_field_names          = hp["normalizer_core_field_names"],
            core_constant_field_names = hp["normalizer_core_constant_field_names"],
        )

        model = cls(
            in_channels       = hp["in_channels"],
            field_names       = hp["field_names"],
            data_shape        = tuple(hp["data_shape"]),
            n_steps_output    = hp["n_steps_output"],
            scale_weights     = hp["scale_weights"],
            vae_cfg           = LatentVAEConfig.from_dict(hp["vae_cfg"]),
            velocity_net_cfg  = VelocityNetConfig.from_dict(hp["velocity_net_cfg"]),
            flow_matching_cfg = FlowMatchingConfig.from_dict(hp["flow_matching_cfg"]),
            optimizer_cfg     = OptimizerConfig.from_dict(hp["optimizer_cfg"]),
            normalizer        = normalizer,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


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
    ) -> torch.Tensor:
        """
        Autoregressive rollout in the VAE latent space.

        Mirrors WaveletFlowModule.rollout exactly.  The three latent-specific
        differences are marked with # LATENT: comments:

        1.  c0 is computed via _to_latent(x0_tiled) instead of
            _to_coeff_dict_x0, to stay consistent with _compute_loss.
        2.  x_tau is a single-scale dict {0: tensor} whose channel count is
            latent_channels * T_out (not in_channels * T_out * n_coeff_i).
        3.  Decode via _from_latent(x_tau, scales=None) — no per-sample
            scale is available at inference; the fixed input_scale in
            _encode_frame / _decode_frame absorbs the rescaling.

        Parameters
        ----------
        seed : (B, Ti, H, W, C)
            Seed frames in physical space (or already z-score normalised if
            ``normalized_input=True``).  Ti must be >= 1.
        scalar_conds : (B, num_cond)
            Scalar conditioning variables, e.g. PDE parameters / Re number.
        rollout_steps : int
            Total output frames to produce.
        n_steps_input : int or None
            Sliding-window width.  Defaults to seed.shape[1].
        n_ensemble : int or None
            Ensemble size E.  Defaults to flow_matching_cfg.n_ensemble.
        normalized_input : bool
            If True, skip normalisation (seed already in z-score space).
        verbose : bool
            Print per-call progress.

        Returns
        -------
        Tensor of shape (B, E, rollout_steps, H, W, C)  — physical space, CPU.
        """
        fm     = self.flow_matching_cfg
        n_ens  = n_ensemble if n_ensemble is not None else fm.n_ensemble
        T_out  = self.n_steps_output
        dt     = 1.0 / fm.num_sampling_steps
        device = self.device
        dtype  = next(self.parameters()).dtype

        # ── Input validation ──────────────────────────────────────────────────
        if seed.ndim != 5:
            raise ValueError(
                f"seed must be (B, Ti, H, W, C), got {tuple(seed.shape)}"
            )
        B, Ti_seed, H, W, C = seed.shape

        Ti = n_steps_input if n_steps_input is not None else Ti_seed
        if Ti_seed < Ti:
            raise ValueError(
                f"seed has only {Ti_seed} frames but n_steps_input={Ti} requested."
            )
        if Ti < 1:
            raise ValueError("n_steps_input must be >= 1.")

        # ── 1. Normalise & trim seed window ───────────────────────────────────
        seed = seed.to(device=device, dtype=dtype)

        seed_norm = (
            seed if normalized_input
            else self.normalizer.normalize_flattened(seed, mode='variable')
        )
        seed_norm = seed_norm[:, -Ti:]          # (B, Ti, H, W, C)

        # ── 2. Expand seed & scalar conds over ensemble members ───────────────
        windows = (
            seed_norm
            .unsqueeze(1)
            .expand(-1, n_ens, -1, -1, -1, -1)
            .reshape(B * n_ens, Ti, H, W, C)
            .clone()                            # (B*E, Ti, H, W, C)
        )

        scalar_conds = scalar_conds.to(device=device, dtype=dtype)
        scalar_exp   = (
            scalar_conds
            .unsqueeze(1)
            .expand(-1, n_ens, -1)
            .reshape(B * n_ens, -1)             # (B*E, num_cond)
        )

        BxE = B * n_ens

        # ── 3. Allocate output buffer on CPU ──────────────────────────────────
        preds_phys = torch.zeros(
            BxE, rollout_steps, H, W, C,
            device='cpu', dtype=dtype,
        )

        # ── LATENT: latent spatial dimensions from the probed shape ──────────
        H_lat  = self.scale_shapes[0]['H']
        W_lat  = self.scale_shapes[0]['W']
        # Channel count that _to_latent packs into the channel axis:
        #   latent_channels (= G * lat_ch_per_group) per frame × T_out frames
        lat_ch_total = self.latent_channels * T_out

        # ── 4. Autoregressive loop ─────────────────────────────────────────────
        step_out   = 0
        call_count = 0
        n_calls    = math.ceil(rollout_steps / T_out)

        while step_out < rollout_steps:
            call_count += 1
            if verbose:
                print(
                    f"  rollout call {call_count:>4d}/{n_calls} "
                    f"(frames {step_out+1}–{min(step_out+T_out, rollout_steps)}"
                    f"/{rollout_steps})",
                    end='\r',
                )

            # ── Conditioning ──────────────────────────────────────────────────
            # curr:  (B*E, C, 1, H, W)     — last frame in z-score space
            # past:  (B*E, C, Ti-1, H, W)  — preceding frames, z-score space
            # (context is always in physical/z-score space, never VAE-encoded)
            curr_norm = rearrange(windows[:, -1:], 'Be 1 H W Cv -> Be Cv 1 H W')
            past_norm = rearrange(windows[:, :-1], 'Be T H W Cv -> Be Cv T H W')

            # LATENT 1: tile current frame to T_out before encoding, matching
            # the _compute_loss convention: x0_tiled = x0.expand(..., T_out, ...)
            x0_tiled = curr_norm.expand(-1, -1, T_out, -1, -1)   # (B*E, C, T_out, H, W)
            c0, _    = self._to_latent(x0_tiled)                 # {0: (B*E, lat_ch*T_out, H_lat, W_lat)}

            # LATENT 2: noise lives in the single latent scale, channel count
            # is latent_channels * T_out (not in_channels * T_out * n_coeff).
            x_tau = {
                0: torch.randn(BxE, lat_ch_total, H_lat, W_lat,
                            device=device, dtype=dtype)
            }

            # ── Euler integration ──────────────────────────────────────────────
            for s in range(fm.num_sampling_steps):
                t_batch = torch.full(
                    (BxE,), s / fm.num_sampling_steps,
                    device=device, dtype=dtype,
                )
                vel = self.net(
                    curr         = c0,
                    x_tau        = x_tau,
                    t            = t_batch,
                    scalar_conds = scalar_exp,
                    context      = past_norm,
                )
                x_tau[0] = x_tau[0] + dt * vel[0]

            # LATENT 3: decode without per-sample scales — the fixed input_scale
            # in _encode_frame / _decode_frame handles the rescaling symmetrically.
            pred_phys = self._from_latent(x_tau, scales=None)    # (B*E, T_out, H, W, C)

            # ── Store (clip to remaining budget) ──────────────────────────────
            frames_to_store = min(T_out, rollout_steps - step_out)
            preds_phys[:, step_out : step_out + frames_to_store] = (
                pred_phys[:, :frames_to_store].cpu()
            )
            step_out += frames_to_store

            # ── Slide context window forward ───────────────────────────────────
            pred_norm = self.normalizer.normalize_flattened(
                pred_phys, mode='variable'
            )                                                       # (B*E, T_out, H, W, C)
            windows = torch.cat([windows, pred_norm], dim=1)[:, -Ti:]

        if verbose:
            print()

        # ── 5. Reshape to (B, E, rollout_steps, H, W, C) ──────────────────────
        return preds_phys.view(B, n_ens, rollout_steps, H, W, C)
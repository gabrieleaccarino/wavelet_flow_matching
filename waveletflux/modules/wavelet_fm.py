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

def _normed_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Variance-normalised MSE.

    var shape (B, C, 1, 1): per-channel spatial variance.
    Normalises across subbands with very different energy levels (LL vs HH).
    """
    var = target.var(dim=(2, 3), keepdim=True).detach().clamp(min=1e-4)
    return F.mse_loss(pred / var.sqrt(), target / var.sqrt())

def _vmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Alternative variance-normalised MSE that normalises each sample and channel
    by its own variance rather than using the batch variance.

    This is more expensive to compute but may be more stable when batch sizes
    are very small or when there are many zero-variance channels.
    """
    l = (pred - target).square().mean(dim=(2, 3)) / (target.var(dim=(2, 3)) + 1e-2)
    l = l.mean()  # average over B and C
    return l

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

        # Store normalizer stats in hparams so they are embedded in the
        # checkpoint.  The normalizer object is not serialisable, so we
        # exclude it and reconstruct it in from_checkpoint() from these
        # plain-dict stats.
        self.save_hyperparameters({
            "in_channels":       in_channels,
            "field_names":       field_names,
            "data_shape":        data_shape,
            "n_steps_output":    n_steps_output,
            "scale_weights":     scale_weights,
            "wavelet_cfg":       wavelet_cfg.to_dict(),
            "velocity_net_cfg":  velocity_net_cfg.to_dict(),   # includes n_levels
            "flow_matching_cfg": flow_matching_cfg.to_dict(),
            "optimizer_cfg":     optimizer_cfg.to_dict(),
            # Normalizer internals — reconstruct on load
            #"normalizer_stats":                     normalizer.stats,
            #"normalizer_means":                     normalizer.means,
            #"normalizer_stds":                      normalizer.stds,
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

        The dummy uses T = n_steps_output (not T=1) so that the probe is
        semantically correct even when n_steps_output > 1.  H_s, W_s, and
        n_coeff do not depend on T, but using the real T avoids any surprise
        if the decomposer has T-dependent paths.
        """
        H, W  = self.data_shape
        dummy = torch.zeros(1, self.in_channels, self.n_steps_output, H, W)

        if self.use_wavelet:
            was = self.wavelet_decomposer.training
            self.wavelet_decomposer.eval()
            coeffs = self.wavelet_decomposer.forward_transform(dummy)
            self.wavelet_decomposer.train(was)
            # coeffs[i] shape: (B, C, T, n_coeff, H_s, W_s)
            return {
                i: {'H': c.shape[-2], 'W': c.shape[-1], 'n_coeff': c.shape[3]}
                for i, c in coeffs.items()
            }
        # Data-space: single scale, full resolution, n_coeff=1
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

        This ensures curr[i] and x_tau[i] always have the same channel count
        C*T*n_coeff_i, which is required by the stem convolutions.
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

        x0 = rearrange(inp[:, -1:], 'B 1 H W C -> B C 1 H W')   # T=1
        x1 = rearrange(out,         'B T H W C -> B C T H W')   # T=n_steps_output
        xp = rearrange(inp[:, :-1], 'B T H W C -> B C T H W')   # past context

        # ── Scalar conditions ──────────────────────────────────────────────────
        scalars = batch['constant_scalars'].to(device) \
                if 'constant_scalars' in batch else None

        # Extract per-trajectory mean dt from the time grid — shape (B, 1)
        dt_cond = None
        if 'input_time_grid' in batch.keys():
            time_grid = batch['input_time_grid'].to(device)   # (B, T)
            dt_cond   = (time_grid[:, 1] - time_grid[:, 0]).unsqueeze(-1)

        # Concatenate whatever conditions are available
        cond_parts = [c for c in (scalars, dt_cond) if c is not None]
        const = torch.cat(cond_parts, dim=-1) if cond_parts else None

        cond = {
            'const'      : const,
            'past_states': xp,
        }
        return x0, x1, cond, x0.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch, log_prefix: str) -> torch.Tensor:
        """
        OT-CFM loss in wavelet space, weighted sum over scales.

        For each scale i:
          ε ~ N(0, I)
          x_tau[i] = t·x1[i] + (1−t)·ε          (OT-CFM interpolant)
          v_target[i] = x1[i] − ε                (OT-CFM velocity)
          loss[i] = normed_mse(vel[i], v_target[i])

        Total loss = Σ_i  (w_i / Σw) · loss[i]

        x0 is tiled to T=n_steps_output via _to_coeff_dict_x0 so that
        curr[i] and x_tau[i] have matching channel dimensions.
        """
        x0, x1, cond, B = self._prepare_batch(batch)

        # T-fix: tile x0 to T so curr[i] matches x_tau[i] channel-wise
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
            #loss_i = _normed_mse(vel[i], targets[i])
            loss_i = _vmse(vel[i], targets[i])
            total  = total + (self.scale_weights[i] / w_sum) * loss_i
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

        x_tau[i] is initialised to pure Gaussian noise at t=0.
        After num_sampling_steps Euler steps x_tau[i] ≈ sample from p(x1).
        IDWT reconstructs the spatial field.

        x0 is tiled to T=n_steps_output via _to_coeff_dict_x0.
        """
        fm    = self.flow_matching_cfg
        n_ens = n_ensemble or fm.n_ensemble
        dt    = 1.0 / fm.num_sampling_steps

        x0, x1, cond, B = self._prepare_batch(batch)

        # T-fix: tile x0 to T
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

            members.append(self._from_coeff_dict(x_tau))    # (B, T, H, W, C)

        yhat = torch.stack(members, dim=1)  # (B, E, T, H, W, C)
        #y    = self._from_coeff_dict(c1)    # (B, T, H, W, C)
        y = self.normalizer.denormalize_flattened(rearrange(x1, 'B C T H W -> B T H W C'), mode='variable')
        return yhat, y


    # ─────────────────────────────────────────────────────────────────────────
    # Autoregressive rollout
    # ─────────────────────────────────────────────────────────────────────────
 
    @torch.no_grad()
    def rollout(
        self,
        seed          : torch.Tensor,
        scalar_conds  : torch.Tensor,
        rollout_steps : int,
        n_steps_input : int | None = None,
        n_ensemble    : int | None = None,
        normalized_input: bool = False,
        verbose       : bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive rollout from an initial seed window.
 
        At each iteration the model predicts ``n_steps_output`` frames; the
        sliding context window is advanced by that many frames and the next
        call uses the freshly predicted frames as conditioning.  The loop
        runs for ``ceil(rollout_steps / n_steps_output)`` model calls and the
        output is trimmed to exactly ``rollout_steps`` frames.
 
        The entire ensemble is evaluated in a single batched forward pass
        (batch dimension = B × N_ENSEMBLE) rather than a sequential member
        loop, keeping GPU utilisation high.
 
        Parameters
        ----------
        seed : Tensor, shape (B, Ti, H, W, C)
            Seed frames in *physical* space (or already z-score normalised if
            ``normalized_input=True``).  ``Ti`` must be >= 1.  If ``Ti``
            exceeds ``n_steps_input``, the last ``n_steps_input`` frames are
            used as the initial window.
        scalar_conds : Tensor, shape (B, num_cond)
            Scalar conditioning variables (e.g. PDE parameters / Reynolds
            number).  Replicated across ensemble members automatically.
        rollout_steps : int
            Total number of output time steps to produce.
        n_steps_input : int or None
            Sliding-window width Ti used for conditioning.  Defaults to
            ``seed.shape[1]`` (use the full seed).  Must match the value used
            during training.
        n_ensemble : int or None
            Ensemble size E.  Defaults to ``flow_matching_cfg.n_ensemble``.
        normalized_input : bool
            If ``True``, ``seed`` is treated as already z-score normalised and
            the normaliser is *not* applied.
        verbose : bool
            If ``True``, print a progress line for each model call.
 
        Returns
        -------
        preds : Tensor, shape (B, E, rollout_steps, H, W, C)
            Ensemble predictions in *physical* (denormalised) space, on CPU.
 
        Notes
        -----
        * The method runs entirely under ``torch.no_grad()`` and moves the
          output to CPU incrementally to avoid OOM on long rollouts.
        * When ``n_steps_output > 1`` the model simultaneously predicts
          multiple frames, so the number of network evaluations per rollout
          step is ``ceil(rollout_steps / n_steps_output)``.
        """
        fm     = self.flow_matching_cfg
        n_ens  = n_ensemble if n_ensemble is not None else fm.n_ensemble
        T_out  = self.n_steps_output      # frames predicted per call
        dt     = 1.0 / fm.num_sampling_steps
        device = self.device
        dtype  = next(self.parameters()).dtype
 
        # ── Input validation ──────────────────────────────────────────────────
        if seed.ndim != 5:
            raise ValueError(
                f"seed must be (B, Ti, H, W, C), got shape {tuple(seed.shape)}"
            )
        B, Ti_seed, H, W, C = seed.shape
 
        Ti = n_steps_input if n_steps_input is not None else Ti_seed
        if Ti_seed < Ti:
            raise ValueError(
                f"seed has only {Ti_seed} frames but n_steps_input={Ti} was requested."
            )
        if Ti < 1:
            raise ValueError("n_steps_input must be >= 1.")
 
        # ── 1. Normalise & trim seed window ───────────────────────────────────
        seed = seed.to(device=device, dtype=dtype)
 
        if not normalized_input:
            seed_norm = self.normalizer.normalize_flattened(seed, mode='variable')
        else:
            seed_norm = seed
 
        seed_norm = seed_norm[:, -Ti:]   # (B, Ti, H, W, C) — keep last Ti frames
 
        # ── 2. Expand seed & scalar conds across ensemble members ─────────────
        # Layout: rows 0..E-1 → batch element 0, rows E..2E-1 → batch element 1, …
        # seed_norm : (B, Ti, H, W, C) → (B, 1, Ti, H, W, C) → (B, E, Ti, H, W, C)
        windows = (
            seed_norm
            .unsqueeze(1)
            .expand(-1, n_ens, -1, -1, -1, -1)
            .reshape(B * n_ens, Ti, H, W, C)
            .clone()                              # (B*E, Ti, H, W, C)
        )
 
        if scalar_conds is not None:
            scalar_conds = scalar_conds.to(device=device, dtype=dtype)
            scalar_exp   = (
                scalar_conds
                .unsqueeze(1)
                .expand(-1, n_ens, -1)
                .reshape(B * n_ens, -1)              # (B*E, num_cond)
            )
        else:
            scalar_exp = None
 
        BxE = B * n_ens
 
        # ── 3. Allocate output buffer on CPU (avoid OOM on long rollouts) ─────
        preds_phys = torch.zeros(
            BxE, rollout_steps, H, W, C,
            device='cpu', dtype=dtype,
        )
 
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
            # curr: last frame of each member's window, shape (B*E, C, 1, H, W)
            # past: all preceding frames,              shape (B*E, C, Ti-1, H, W)
            curr_norm = rearrange(windows[:, -1:], 'B 1 H W Cv -> B Cv 1 H W')
            past_norm = rearrange(windows[:, :-1], 'B T H W Cv -> B Cv T H W')
 
            # c0[i]: (B*E, C·T_out·n_coeff_i, H_s, W_s)  (tiled to T_out internally)
            c0 = self._to_coeff_dict_x0(curr_norm)
 
            # ── Initialise noise in wavelet coefficient space ──────────────────
            # Each scale has its own spatial resolution and coefficient count.
            x_tau = {
                i: torch.randn(
                    BxE,
                    self.in_channels * T_out * self.scale_shapes[i]['n_coeff'],
                    self.scale_shapes[i]['H'],
                    self.scale_shapes[i]['W'],
                    device=device,
                    dtype=dtype,
                )
                for i in range(self.n_scales)
            }
 
            # ── Euler integration ──────────────────────────────────────────────
            for s in range(fm.num_sampling_steps):
                t_batch = torch.full(
                    (BxE,),
                    s / fm.num_sampling_steps,
                    device=device,
                    dtype=dtype,
                )
                vel = self.net(
                    curr        = c0,
                    x_tau       = x_tau,
                    t           = t_batch,
                    scalar_conds= scalar_exp,
                    context     = past_norm,
                )
                for i in range(self.n_scales):
                    x_tau[i] = x_tau[i] + dt * vel[i]
 
            # ── Decode: (B*E, T_out, H, W, C) in physical space ───────────────
            pred_phys = self._from_coeff_dict(x_tau)   # (B*E, T_out, H, W, C)
 
            # ── Store (clip to remaining budget) ──────────────────────────────
            frames_to_store = min(T_out, rollout_steps - step_out)
            preds_phys[:, step_out : step_out + frames_to_store] = (
                pred_phys[:, :frames_to_store].cpu()
            )
            step_out += frames_to_store
 
            # ── Slide the context window forward by T_out frames ───────────────
            # Normalise the *full* T_out prediction before appending so that
            # the window always contains z-score normalised values.
            pred_norm = self.normalizer.normalize_flattened(
                pred_phys, mode='variable'
            )                                                        # (B*E, T_out, H, W, C)
 
            # Concatenate new frames and keep only the last Ti frames.
            windows = torch.cat([windows, pred_norm], dim=1)[:, -Ti:]
 
        if verbose:
            print()   # newline after \r progress
 
        # ── 5. Reshape to (B, E, rollout_steps, H, W, C) ──────────────────────
        return preds_phys.view(B, n_ens, rollout_steps, H, W, C)
 

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
        """
        Parameters
        ----------
        ensemble_reduce : how to collapse the E dimension of yhat before
            plotting.
            'random' — pick one member at random (default)
            'mean'   — average across all members in physical space
        """
        if self.experiment_manager is None:
            return
        should_save = (
            self.current_epoch % self.experiment_manager.log_every_n_epochs == 0
            or self.current_epoch == self.trainer.max_epochs - 1
        )
        if not should_save:
            return

        batch, batch_idx = self._get_random_validation_sample()
        yhat, y = self.sample(batch, batch_idx)   # yhat: (B, E, T, H, W, C)

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

    #@classmethod
    #def from_checkpoint(cls, path: str, map_location="cpu") -> "WaveletFlowModule":
    #    """
    #    Load a fully self-contained checkpoint.
    #    No TOML config, no distributed setup — just the .ckpt file.

    #    VelocityNetConfig.from_dict must handle the 'n_levels' key that is
    #    now included in the saved velocity_net_cfg dict.

    #    Usage:
    #        model = WaveletFlowModule.from_checkpoint("run.ckpt")
    #        model.eval()
    #    """
    #    from the_well.data.normalization import ZScoreNormalization
    #    from waveletflux.utils.configs import (
    #        WaveletConfig, VelocityNetConfig, FlowMatchingConfig, OptimizerConfig,
    #    )

    #    ckpt = torch.load(path, map_location=map_location)
    #    hp   = ckpt["hyper_parameters"]

    #    means = {'mean': hp["normalizer_means"]}
    #    stds  = {'std' : hp["normalizer_stds"]}

    #    normalizer = ZScoreNormalization(
    #        stats                     = means | stds,
    #        core_field_names          = hp["normalizer_core_field_names"],
    #        core_constant_field_names = hp["normalizer_core_constant_field_names"],
    #    )

    #    model = cls(
    #        in_channels       = hp["in_channels"],
    #        field_names       = hp["field_names"],
    #        data_shape        = tuple(hp["data_shape"]),
    #        n_steps_output    = hp["n_steps_output"],
    #        scale_weights     = hp["scale_weights"],
    #        wavelet_cfg       = WaveletConfig.from_dict(hp["wavelet_cfg"]),
    #        velocity_net_cfg  = VelocityNetConfig.from_dict(hp["velocity_net_cfg"]),
    #        flow_matching_cfg = FlowMatchingConfig.from_dict(hp["flow_matching_cfg"]),
    #        optimizer_cfg     = OptimizerConfig.from_dict(hp["optimizer_cfg"]),
    #        normalizer        = normalizer,
    #    )
    #    model.load_state_dict(ckpt["state_dict"])
    #    return model
    
    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu", stats: dict = None) -> "WaveletFlowModule":
        from the_well.data.normalization import ZScoreNormalization
        from waveletflux.utils.configs import (
            WaveletConfig, VelocityNetConfig, FlowMatchingConfig, OptimizerConfig,
        )

        ckpt = torch.load(path, map_location=map_location)
        hp   = ckpt["hyper_parameters"]

        # ── Normalizer stats: external override > new format > old partial format ──
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
            wavelet_cfg       = WaveletConfig.from_dict(hp["wavelet_cfg"]),
            velocity_net_cfg  = VelocityNetConfig.from_dict(hp["velocity_net_cfg"]),
            flow_matching_cfg = FlowMatchingConfig.from_dict(hp["flow_matching_cfg"]),
            optimizer_cfg     = OptimizerConfig.from_dict(hp["optimizer_cfg"]),
            normalizer        = normalizer,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model
    
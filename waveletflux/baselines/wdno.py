import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from waveletflux.models.wavelet_unet import WaveletFlowUNet
from waveletflux.wavelets.DWT2D import WaveletDecomposer2D


def _extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (x.ndim - 1))).to(device=x.device, dtype=x.dtype)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999).to(torch.float32)


def _linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 1e-4
    beta_end = scale * 2e-2
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class WDNO2d(nn.Module):
    """
    Target-only wavelet diffusion baseline with clean conditioning.

    The model diffuses only the next-frame wavelet coefficients and conditions
    on:
    - the most recent observed frame (`curr`)
    - the remaining past frames (`context`)
    - scalar conditions (`constant_scalars + dt`)

    This matches the repo's `4 -> 1` setup more closely than the upstream
    full-trajectory formulation while avoiding mixed observed/generated states.
    """

    def __init__(
        self,
        n_fields: int,
        n_steps_input: int,
        n_steps_output: int,
        size: Sequence[int],
        hidden_channels: int = 64,
        n_res: int = 2,
        n_levels: int = 3,
        n_scales: int = 3,
        cond_dim: int = 128,
        wavelet: str = "db2",
        mode: str = "periodic",
        scalar_cond_dim: int = 0,
        diffusion_steps: int = 100,
        sampling_steps: int = 100,
        beta_schedule: str = "cosine",
        ddim_eta: float = 1.0,
        x_start_clip: float = 4.0,
    ):
        super().__init__()

        if len(size) != 2:
            raise ValueError(f"`size` must have length 2, got {tuple(size)}.")
        if n_fields < 1 or n_steps_input < 1 or n_steps_output < 1:
            raise ValueError("`n_fields`, `n_steps_input`, and `n_steps_output` must be >= 1.")
        if n_steps_output != 1:
            raise ValueError(f"WDNO2d currently supports `n_steps_output=1` only, got {n_steps_output}.")
        if n_scales < 1:
            raise ValueError(f"`n_scales` must be >= 1, got {n_scales}.")
        if diffusion_steps < 2:
            raise ValueError(f"`diffusion_steps` must be >= 2, got {diffusion_steps}.")
        if sampling_steps < 1:
            raise ValueError(f"`sampling_steps` must be >= 1, got {sampling_steps}.")

        self.n_fields = int(n_fields)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.size = (int(size[0]), int(size[1]))
        self.hidden_channels = int(hidden_channels)
        self.n_res = int(n_res)
        self.n_levels = int(n_levels)
        self.n_scales = int(n_scales)
        self.cond_dim = int(cond_dim)
        self.wavelet = wavelet
        self.mode = mode
        self.scalar_cond_dim = int(scalar_cond_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.sampling_steps = int(sampling_steps)
        self.beta_schedule = beta_schedule
        self.ddim_eta = float(ddim_eta)
        self.x_start_clip = float(x_start_clip) if x_start_clip is not None else None

        self.in_channels = self.n_fields * self.n_steps_input
        self.out_channels = self.n_fields * self.n_steps_output
        self.n_context_frames = max(self.n_steps_input - 1, 0)

        self.wavelet_decomposer = WaveletDecomposer2D(
            wavelet=self.wavelet,
            n_scales=self.n_scales,
            mode=self.mode,
        )
        self.scale_shapes = self._probe_wavelet_shapes()

        for i in range(self.n_scales):
            wav_ch = self.n_fields * self.n_steps_output * self.scale_shapes[i]["n_coeff"]
            self.register_buffer(
                f"coeff_scale_{i}",
                torch.ones(1, wav_ch, 1, 1, dtype=torch.float32),
            )

        self.net = WaveletFlowUNet(
            scale_shapes=self.scale_shapes,
            in_channels=self.n_fields,
            n_steps_output=self.n_steps_output,
            n_scales=self.n_scales,
            n_levels=self.n_levels,
            context_channels=self.n_fields if self.n_context_frames > 0 else 0,
            n_context_frames=self.n_context_frames,
            cond_dim=self.cond_dim,
            num_scalar_conds=self.scalar_cond_dim,
            base_ch=self.hidden_channels,
            n_res=self.n_res,
        )

        if self.beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.diffusion_steps)
        elif self.beta_schedule == "linear":
            betas = _linear_beta_schedule(self.diffusion_steps)
        else:
            raise ValueError(
                f"Unknown beta schedule `{self.beta_schedule}`. Expected `cosine` or `linear`."
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @torch.no_grad()
    def _probe_wavelet_shapes(self) -> Dict[int, dict]:
        dummy = torch.zeros(1, self.n_fields, self.n_steps_output, *self.size)
        was_training = self.wavelet_decomposer.training
        self.wavelet_decomposer.eval()
        coeffs = self.wavelet_decomposer.forward_transform(dummy)
        self.wavelet_decomposer.train(was_training)
        return {
            i: {"H": c.shape[-2], "W": c.shape[-1], "n_coeff": c.shape[3]}
            for i, c in coeffs.items()
        }

    def _check_scalar_conds(
        self,
        scalar_conds: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.scalar_cond_dim == 0:
            if scalar_conds is not None and scalar_conds.shape[-1] != 0:
                raise ValueError(
                    f"Model was configured with `scalar_cond_dim=0`, got {scalar_conds.shape[-1]} scalar conditions."
                )
            return None

        if scalar_conds is None:
            raise ValueError(f"This WDNO expects {self.scalar_cond_dim} scalar conditions per sample.")
        if scalar_conds.ndim != 2 or scalar_conds.shape[0] != batch_size:
            raise ValueError(
                "Expected scalar conditions with shape `(B, S)`, got "
                f"{tuple(scalar_conds.shape)}."
            )
        if scalar_conds.shape[1] != self.scalar_cond_dim:
            raise ValueError(
                f"Expected {self.scalar_cond_dim} scalar conditions, got {scalar_conds.shape[1]}."
            )
        return scalar_conds.to(device=device, dtype=dtype)

    def _flat_to_fields(self, flat: torch.Tensor, n_steps: int) -> torch.Tensor:
        if flat.ndim != 4:
            raise ValueError(f"Expected flat tensor `(B, C, H, W)`, got {tuple(flat.shape)}.")
        batch_size, channels, height, width = flat.shape
        expected = self.n_fields * n_steps
        if channels != expected:
            raise ValueError(f"Expected {expected} channels, got {channels}.")
        return flat.reshape(batch_size, n_steps, self.n_fields, height, width).permute(0, 1, 3, 4, 2)

    def _fields_to_flat(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim != 5:
            raise ValueError(f"Expected fields `(B, T, H, W, F)`, got {tuple(fields.shape)}.")
        return rearrange(fields, "B T H W F -> B (T F) H W")

    def _to_coeff_dict(self, fields: torch.Tensor) -> Dict[int, torch.Tensor]:
        if fields.ndim != 5:
            raise ValueError(f"Expected fields `(B, T, H, W, F)`, got {tuple(fields.shape)}.")
        spatial = rearrange(fields, "B T H W F -> B F T H W")
        raw = self.wavelet_decomposer.forward_transform(spatial)
        return {
            i: rearrange(v, "B C T Coeff H W -> B (C T Coeff) H W")
            for i, v in raw.items()
        }

    def _to_coeff_dict_x0(self, current_fields: torch.Tensor) -> Dict[int, torch.Tensor]:
        tiled = current_fields.expand(-1, -1, -1, -1, -1)
        return self._to_coeff_dict(tiled)

    def _from_coeff_dict(self, coeff: Dict[int, torch.Tensor]) -> torch.Tensor:
        raw = {
            i: rearrange(
                v,
                "B (C T Coeff) H W -> B C T Coeff H W",
                C=self.n_fields,
                T=self.n_steps_output,
                Coeff=self.scale_shapes[i]["n_coeff"],
            )
            for i, v in coeff.items()
        }
        spatial = self.wavelet_decomposer.inverse_transform(raw)
        return rearrange(spatial, "B C T H W -> B T H W C")

    def _normalize_coeff_dict(self, coeff: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        out = {}
        for i, v in coeff.items():
            scale = getattr(self, f"coeff_scale_{i}").to(device=v.device, dtype=v.dtype)
            out[i] = v / scale
        return out

    def _denormalize_coeff_dict(self, coeff: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        out = {}
        for i, v in coeff.items():
            scale = getattr(self, f"coeff_scale_{i}").to(device=v.device, dtype=v.dtype)
            out[i] = v * scale
        return out

    def _coeff_dict_from_supervised_tensors(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        y_fields = self._flat_to_fields(y, self.n_steps_output)
        return self._to_coeff_dict(y_fields)

    def set_coeff_scales(self, coeff_scales: Dict[int, torch.Tensor]) -> None:
        for i in range(self.n_scales):
            scale = coeff_scales.get(i)
            if scale is None:
                raise KeyError(f"Expected coefficient scale entry for key {i}.")
            ref = getattr(self, f"coeff_scale_{i}")
            if scale.shape != ref.shape:
                raise ValueError(
                    f"Scale tensor for key {i} has shape {tuple(scale.shape)}, expected {tuple(ref.shape)}."
                )
            ref.copy_(scale.to(device=ref.device, dtype=ref.dtype))

    def _conditioning(
        self,
        x: torch.Tensor,
        scalar_conds: Optional[torch.Tensor],
    ):
        batch_size = x.shape[0]
        scalar_conds = self._check_scalar_conds(
            scalar_conds,
            batch_size=batch_size,
            device=x.device,
            dtype=torch.float32,
        )

        observed_fields = self._flat_to_fields(x, self.n_steps_input)
        curr_fields = observed_fields[:, -1:]
        context = None
        if self.n_context_frames > 0:
            context = rearrange(observed_fields[:, :-1], "B T H W F -> B F T H W")
        return curr_fields, context, scalar_conds

    def _q_sample(
        self,
        x_start: Dict[int, torch.Tensor],
        t: torch.Tensor,
        noise: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        return {
            i: _extract(self.sqrt_alphas_cumprod, t, x_start[i]) * x_start[i]
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start[i]) * noise[i]
            for i in x_start
        }

    def _predict_noise(
        self,
        curr: Dict[int, torch.Tensor],
        state: Dict[int, torch.Tensor],
        t: torch.Tensor,
        scalar_conds: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        t_norm = t.float() / max(self.diffusion_steps - 1, 1)
        return self.net(curr=curr, x_tau=state, t=t_norm, scalar_conds=scalar_conds, context=context)

    def _clip_coeff_dict(self, coeff: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        if self.x_start_clip is None:
            return coeff
        return {i: v.clamp(-self.x_start_clip, self.x_start_clip) for i, v in coeff.items()}

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(
                f"Expected flat inputs `(B, C, H, W)`, got {tuple(x.shape)} and {tuple(y.shape)}."
            )

        curr_fields, context, scalar_conds = self._conditioning(x, scalar_conds)
        target_fields = self._flat_to_fields(y, self.n_steps_output)

        curr = self._normalize_coeff_dict(self._to_coeff_dict_x0(curr_fields))
        target = self._normalize_coeff_dict(self._to_coeff_dict(target_fields))

        batch_size = x.shape[0]
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=x.device).long()
        noise = {i: torch.randn_like(v) for i, v in target.items()}
        state = self._q_sample(x_start=target, t=t, noise=noise)
        pred_noise = self._predict_noise(curr, state, t, scalar_conds, context)

        losses = [F.mse_loss(pred_noise[i], noise[i]) for i in target]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample_fields(
        self,
        x: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        curr_fields, context, scalar_conds = self._conditioning(x, scalar_conds)
        curr = self._normalize_coeff_dict(self._to_coeff_dict_x0(curr_fields))

        batch_size = x.shape[0]
        sample = {
            i: torch.randn(
                batch_size,
                self.n_fields * self.n_steps_output * self.scale_shapes[i]["n_coeff"],
                self.scale_shapes[i]["H"],
                self.scale_shapes[i]["W"],
                device=x.device,
                dtype=torch.float32,
            )
            for i in range(self.n_scales)
        }

        step_ids = torch.linspace(
            self.diffusion_steps - 1,
            0,
            self.sampling_steps,
            device=x.device,
        ).round().long()
        step_ids = torch.unique_consecutive(step_ids)
        next_ids = torch.cat([step_ids[1:], step_ids.new_tensor([-1])], dim=0)

        for step, next_step in zip(step_ids, next_ids):
            t = torch.full((batch_size,), int(step.item()), device=x.device, dtype=torch.long)
            pred_noise = self._predict_noise(curr, sample, t, scalar_conds, context)

            x_start = {
                i: (
                    sample[i] - _extract(self.sqrt_one_minus_alphas_cumprod, t, sample[i]) * pred_noise[i]
                ) / _extract(self.sqrt_alphas_cumprod, t, sample[i])
                for i in sample
            }
            x_start = self._clip_coeff_dict(x_start)
            pred_noise = {
                i: (
                    sample[i] - _extract(self.sqrt_alphas_cumprod, t, sample[i]) * x_start[i]
                ) / _extract(self.sqrt_one_minus_alphas_cumprod, t, sample[i])
                for i in sample
            }

            if next_step.item() < 0:
                sample = x_start
                break

            alpha_next = self.alphas_cumprod[int(next_step.item())].to(device=x.device)
            if self.ddim_eta == 0.0:
                sigma = 0.0
                coeff = torch.sqrt(1.0 - alpha_next)
            else:
                alpha_t = self.alphas_cumprod[int(step.item())].to(device=x.device)
                sigma = self.ddim_eta * torch.sqrt(
                    (1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next)
                )
                coeff = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma**2, min=0.0))

            sample = {
                i: (
                    torch.sqrt(alpha_next).to(dtype=x_start[i].dtype) * x_start[i]
                    + coeff.to(dtype=pred_noise[i].dtype) * pred_noise[i]
                    + (
                        sigma.to(device=x.device, dtype=x_start[i].dtype) * torch.randn_like(sample[i])
                        if self.ddim_eta > 0.0
                        else 0.0
                    )
                )
                for i in sample
            }

        coeff_phys = self._denormalize_coeff_dict(sample)
        return self._from_coeff_dict(coeff_phys)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fields = self.sample_fields(x, scalar_conds=scalar_conds)
        return self._fields_to_flat(fields)

    @torch.no_grad()
    def forward_fields(
        self,
        fields: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if fields.ndim != 5:
            raise ValueError(f"Expected input shape `(B, Ti, H, W, F)`, got {tuple(fields.shape)}.")
        return self.sample_fields(self._fields_to_flat(fields), scalar_conds=scalar_conds)


__all__ = ["WDNO2d"]

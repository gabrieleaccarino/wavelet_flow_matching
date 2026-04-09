import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from waveletflux.models.wavelet_unet import WaveletFlowUNet
from waveletflux.wavelets.DWT2D import DiscreteWaveletTransform2D


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
    """Wavelet Diffusion Neural Operator with the WaveletFlux I/O contract."""

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
        wavelet: str = "db6",
        mode: str = "symmetric",
        scalar_cond_dim: int = 0,
        diffusion_steps: int = 100,
        sampling_steps: int = 20,
        beta_schedule: str = "cosine",
        ddim_eta: float = 0.0,
    ):
        super().__init__()

        if len(size) != 2:
            raise ValueError(f"`size` must have length 2, got {tuple(size)}.")
        if n_fields < 1 or n_steps_input < 1 or n_steps_output < 1:
            raise ValueError("`n_fields`, `n_steps_input`, and `n_steps_output` must be >= 1.")
        if n_scales < 1:
            raise ValueError(f"`n_scales` must be >= 1, got {n_scales}.")
        if n_levels < n_scales:
            raise ValueError(
                f"`n_levels` must be >= `n_scales`, got {n_levels} < {n_scales}."
            )
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

        self.in_channels = self.n_fields * self.n_steps_input
        self.out_channels = self.n_fields * self.n_steps_output

        self.wavelet_transform = DiscreteWaveletTransform2D(
            wavelet=self.wavelet,
            mode=self.mode,
            n_scales=self.n_scales,
        )
        self.scale_shapes = self._probe_wavelet_shapes()
        self.net = WaveletFlowUNet(
            scale_shapes=self.scale_shapes,
            in_channels=self.n_fields,
            n_steps_output=self.n_steps_output,
            n_scales=self.n_scales,
            n_levels=self.n_levels,
            context_channels=self.n_fields,
            n_context_frames=self.n_steps_input,
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
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def _probe_wavelet_shapes(self) -> Dict[int, dict]:
        dummy = torch.zeros(1, self.n_fields, self.n_steps_output, *self.size)
        raw = self.wavelet_transform.forward_transform(dummy)
        return {
            i: {"H": int(v.shape[-2]), "W": int(v.shape[-1]), "n_coeff": int(v.shape[3])}
            for i, v in raw.items()
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
                    f"Model was configured with `scalar_cond_dim=0`, got "
                    f"{scalar_conds.shape[-1]} scalar conditions."
                )
            return None

        if scalar_conds is None:
            raise ValueError(
                f"This WDNO expects {self.scalar_cond_dim} scalar conditions per sample."
            )
        if scalar_conds.ndim != 2 or scalar_conds.shape[0] != batch_size:
            raise ValueError(
                "Expected scalar conditions with shape `(B, S)`, got "
                f"{tuple(scalar_conds.shape)}."
            )
        if scalar_conds.shape[1] != self.scalar_cond_dim:
            raise ValueError(
                f"Expected {self.scalar_cond_dim} scalar conditions, got "
                f"{scalar_conds.shape[1]}."
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
            raise ValueError(
                f"Expected fields with shape `(B, T, H, W, F)`, got {tuple(fields.shape)}."
            )
        return rearrange(fields, "B T H W F -> B (T F) H W")

    def _coeff_dict_from_fields(self, fields: torch.Tensor) -> Dict[int, torch.Tensor]:
        # ptwt does not support bf16/autocast, so keep the wavelet transform in fp32.
        spatial = rearrange(fields, "B T H W F -> B F T H W").to(dtype=torch.float32)
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            raw = self.wavelet_transform.forward_transform(spatial)
        return {
            i: rearrange(v, "B C T Coeff H W -> B (C T Coeff) H W")
            for i, v in raw.items()
        }

    def _coeff_dict_to_fields(self, coeffs: Dict[int, torch.Tensor]) -> torch.Tensor:
        raw = {
            i: rearrange(
                v.to(dtype=torch.float32),
                "B (C T Coeff) H W -> B C T Coeff H W",
                C=self.n_fields,
                T=self.n_steps_output,
                Coeff=self.scale_shapes[i]["n_coeff"],
            )
            for i, v in coeffs.items()
        }
        with torch.autocast(device_type=next(iter(raw.values())).device.type, enabled=False):
            spatial = self.wavelet_transform.inverse_transform(raw)
        return rearrange(spatial, "B F T H W -> B T H W F")

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
            dtype=x.dtype,
        )

        x_fields = self._flat_to_fields(x, self.n_steps_input)
        context = rearrange(x_fields, "B Ti H W F -> B F Ti H W").to(dtype=torch.float32)
        curr_fields = x_fields[:, -1:, ...].expand(-1, self.n_steps_output, -1, -1, -1)
        curr = self._coeff_dict_from_fields(curr_fields)
        return curr, context, scalar_conds

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

        curr, context, scalar_conds = self._conditioning(x, scalar_conds)
        target = self._coeff_dict_from_fields(self._flat_to_fields(y, self.n_steps_output))

        batch_size = x.shape[0]
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=x.device).long()
        tau = t.float() / float(self.diffusion_steps - 1)
        noise = {i: torch.randn_like(target_i) for i, target_i in target.items()}
        noisy = self._q_sample(target, t, noise)

        pred = self.net(
            curr=curr,
            x_tau=noisy,
            t=tau,
            scalar_conds=scalar_conds,
            context=context,
        )
        losses = [F.mse_loss(pred[i], noise[i]) for i in pred]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def sample_fields(
        self,
        x: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        curr, context, scalar_conds = self._conditioning(x, scalar_conds)
        sample = {
            i: torch.randn(
                x.shape[0],
                self.n_fields * self.n_steps_output * self.scale_shapes[i]["n_coeff"],
                self.scale_shapes[i]["H"],
                self.scale_shapes[i]["W"],
                device=x.device,
                dtype=torch.float32,
            )
            for i in self.scale_shapes
        }

        step_ids = torch.linspace(
            self.diffusion_steps - 1,
            0,
            self.sampling_steps,
            device=x.device,
        ).round().long()
        step_ids = torch.unique_consecutive(step_ids)
        next_ids = torch.cat(
            [step_ids[1:], step_ids.new_tensor([-1])],
            dim=0,
        )

        for step, next_step in zip(step_ids, next_ids):
            t = torch.full((x.shape[0],), int(step.item()), device=x.device, dtype=torch.long)
            tau = t.float() / float(self.diffusion_steps - 1)
            pred_noise = self.net(
                curr=curr,
                x_tau=sample,
                t=tau,
                scalar_conds=scalar_conds,
                context=context,
            )

            x_start = {
                i: (
                    sample[i] - _extract(self.sqrt_one_minus_alphas_cumprod, t, sample[i]) * pred_noise[i]
                ) / _extract(self.sqrt_alphas_cumprod, t, sample[i])
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
                i: torch.sqrt(alpha_next).to(dtype=x_start[i].dtype) * x_start[i]
                + coeff.to(dtype=pred_noise[i].dtype) * pred_noise[i]
                + (
                    sigma.to(device=x.device, dtype=x_start[i].dtype) * torch.randn_like(sample[i])
                    if self.ddim_eta > 0.0
                    else 0.0
                )
                for i in sample
            }

        return self._coeff_dict_to_fields(sample)

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
            raise ValueError(
                f"Expected input shape `(B, Ti, H, W, F)`, got {tuple(fields.shape)}."
            )
        return self.sample_fields(self._fields_to_flat(fields), scalar_conds=scalar_conds)


__all__ = ["WDNO2d"]

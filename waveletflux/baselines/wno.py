import math
from typing import Optional, Sequence, Tuple

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    DWTForward = None
    DWTInverse = None
    _WAVELET_IMPORT_ERROR = exc
else:
    _WAVELET_IMPORT_ERROR = None


def _require_wavelet_backend() -> None:
    if DWTForward is None or DWTInverse is None:
        raise ImportError(
            "WNO requires `pytorch_wavelets`. Install it with "
            "`python -m pip install --user pytorch_wavelets`."
        ) from _WAVELET_IMPORT_ERROR


class WaveConv2d(nn.Module):
    """Wavelet-domain convolution used by WNO.

    This follows the reference WNO implementation closely:
    - decompose to the coarsest wavelet level,
    - learn separate weights for the final LL/LH/HL/HH coefficients,
    - zero all intermediate detail levels,
    - reconstruct with the inverse DWT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        level: int,
        size: Sequence[int],
        wavelet: str = "db6",
        mode: str = "symmetric",
    ):
        super().__init__()
        _require_wavelet_backend()

        if len(size) != 2:
            raise ValueError(f"`size` must have length 2, got {tuple(size)}.")
        if level < 1:
            raise ValueError(f"`level` must be >= 1, got {level}.")
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Unknown wavelet `{wavelet}`.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.level = int(level)
        self.size = (int(size[0]), int(size[1]))
        self.wavelet = wavelet
        self.mode = mode

        self._base_dwt = DWTForward(J=self.level, mode=self.mode, wave=self.wavelet)
        self._base_idwt = DWTInverse(mode=self.mode, wave=self.wavelet)

        dummy = torch.randn(1, 1, *self.size)
        yl, _ = self._base_dwt(dummy)
        self.coeff_size = (int(yl.shape[-2]), int(yl.shape[-1]))

        scale = 1.0 / max(self.in_channels * self.out_channels, 1)
        coeff_h, coeff_w = self.coeff_size
        self.weights1 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, coeff_h, coeff_w)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, coeff_h, coeff_w)
        )
        self.weights3 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, coeff_h, coeff_w)
        )
        self.weights4 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, coeff_h, coeff_w)
        )

    @staticmethod
    def mul2d(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def _resolve_levels(self, spatial_shape: Tuple[int, int]) -> int:
        if tuple(spatial_shape) == self.size:
            return self.level

        shifts = []
        for actual, base in zip(spatial_shape, self.size):
            ratio = actual / base
            if ratio >= 1.0:
                shift_f = math.log2(ratio)
            else:
                shift_f = -math.log2(base / actual)

            shift = round(shift_f)
            if not math.isclose(shift_f, shift, abs_tol=1e-6):
                raise ValueError(
                    "WNO only supports same-resolution inputs or power-of-two "
                    f"super-resolution relative to the base size {self.size}. "
                    f"Got input size {tuple(spatial_shape)}."
                )
            shifts.append(int(shift))

        if shifts[0] != shifts[1]:
            raise ValueError(
                "WNO requires the same power-of-two scaling factor on both axes. "
                f"Base size: {self.size}, input size: {tuple(spatial_shape)}."
            )

        levels = self.level + shifts[0]
        if levels < 1:
            raise ValueError(
                f"Resolved wavelet level must be >= 1, got {levels} for input "
                f"size {tuple(spatial_shape)}."
            )
        return levels

    def _get_ops(self, x: torch.Tensor) -> Tuple[nn.Module, nn.Module]:
        levels = self._resolve_levels((x.shape[-2], x.shape[-1]))
        if levels == self.level:
            dwt = self._base_dwt
            idwt = self._base_idwt
        else:
            dwt = DWTForward(J=levels, mode=self.mode, wave=self.wavelet)
            idwt = DWTInverse(mode=self.mode, wave=self.wavelet)

        dwt = dwt.to(device=x.device, dtype=x.dtype)
        idwt = idwt.to(device=x.device, dtype=x.dtype)
        return dwt, idwt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input `(B, C, H, W)`, got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}."
            )

        dwt, idwt = self._get_ops(x)
        yl, yh = dwt(x)

        if tuple(yl.shape[-2:]) != self.coeff_size:
            raise ValueError(
                "Resolved coarsest wavelet coefficient size does not match the "
                f"learned kernel shape. Expected {self.coeff_size}, got "
                f"{tuple(yl.shape[-2:])}."
            )

        out_yl = self.mul2d(yl, self.weights1)
        out_yh = [torch.zeros_like(coeff) for coeff in yh]

        out_yh[-1][:, :, 0, :, :] = self.mul2d(yh[-1][:, :, 0, :, :], self.weights2)
        out_yh[-1][:, :, 1, :, :] = self.mul2d(yh[-1][:, :, 1, :, :], self.weights3)
        out_yh[-1][:, :, 2, :, :] = self.mul2d(yh[-1][:, :, 2, :, :], self.weights4)

        out = idwt((out_yl, out_yh))
        return out[..., : x.shape[-2], : x.shape[-1]]


class WNO2d(nn.Module):
    """Faithful 2D Wavelet Neural Operator adapted to WaveletFlux layouts."""

    def __init__(
        self,
        n_fields: int,
        n_steps_input: int,
        n_steps_output: int,
        size: Sequence[int],
        hidden_channels: int = 32,
        n_layers: int = 4,
        level: int = 4,
        wavelet: str = "db6",
        mode: str = "symmetric",
        scalar_cond_dim: int = 0,
        grid_range: Sequence[float] = (1.0, 1.0),
        padding: int = 0,
        fc_channels: int = 128,
    ):
        super().__init__()

        if len(size) != 2:
            raise ValueError(f"`size` must have length 2, got {tuple(size)}.")
        if len(grid_range) != 2:
            raise ValueError(
                f"`grid_range` must have length 2, got {tuple(grid_range)}."
            )
        if n_fields < 1 or n_steps_input < 1 or n_steps_output < 1:
            raise ValueError("`n_fields`, `n_steps_input`, and `n_steps_output` must be >= 1.")

        self.n_fields = int(n_fields)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.size = (int(size[0]), int(size[1]))
        self.hidden_channels = int(hidden_channels)
        self.n_layers = int(n_layers)
        self.level = int(level)
        self.wavelet = wavelet
        self.mode = mode
        self.scalar_cond_dim = int(scalar_cond_dim)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.padding = int(padding)
        self.fc_channels = int(fc_channels)

        self.in_channels = self.n_fields * self.n_steps_input
        self.out_channels = self.n_fields * self.n_steps_output
        lifted_channels = self.in_channels + self.scalar_cond_dim + 2

        self.fc0 = nn.Linear(lifted_channels, self.hidden_channels)
        self.conv = nn.ModuleList(
            [
                WaveConv2d(
                    self.hidden_channels,
                    self.hidden_channels,
                    level=self.level,
                    size=self.size,
                    wavelet=self.wavelet,
                    mode=self.mode,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.w = nn.ModuleList(
            [nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1) for _ in range(self.n_layers)]
        )
        self.fc1 = nn.Linear(self.hidden_channels, self.fc_channels)
        self.fc2 = nn.Linear(self.fc_channels, self.out_channels)

    def _get_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        grid_x = torch.linspace(0.0, self.grid_range[0], height, device=device, dtype=dtype)
        grid_x = grid_x.view(1, 1, height, 1).expand(batch_size, 1, height, width)

        grid_y = torch.linspace(0.0, self.grid_range[1], width, device=device, dtype=dtype)
        grid_y = grid_y.view(1, 1, 1, width).expand(batch_size, 1, height, width)

        return torch.cat((grid_x, grid_y), dim=1)

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
                f"This WNO expects {self.scalar_cond_dim} scalar conditions per sample."
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

    def forward(
        self,
        x: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape `(B, C, H, W)`, got {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )

        batch_size, _, height, width = x.shape
        grid = self._get_grid(batch_size, height, width, x.device, x.dtype)
        pieces = [x, grid]

        scalar_conds = self._check_scalar_conds(
            scalar_conds, batch_size=batch_size, device=x.device, dtype=x.dtype
        )
        if scalar_conds is not None:
            scalar_map = scalar_conds.view(batch_size, self.scalar_cond_dim, 1, 1)
            scalar_map = scalar_map.expand(-1, -1, height, width)
            pieces.append(scalar_map)

        x = torch.cat(pieces, dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        for idx, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if idx != self.n_layers - 1:
                x = F.mish(x)

        if self.padding:
            x = x[..., :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

    def forward_fields(
        self,
        fields: torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if fields.ndim != 5:
            raise ValueError(
                f"Expected input shape `(B, Ti, H, W, F)`, got {tuple(fields.shape)}."
            )

        batch_size, steps_in, height, width, n_fields = fields.shape
        if steps_in != self.n_steps_input:
            raise ValueError(
                f"Expected {self.n_steps_input} input steps, got {steps_in}."
            )
        if n_fields != self.n_fields:
            raise ValueError(f"Expected {self.n_fields} fields, got {n_fields}.")

        flat = fields.permute(0, 1, 4, 2, 3).reshape(batch_size, self.in_channels, height, width)
        out = self(flat, scalar_conds=scalar_conds)
        out = out.reshape(batch_size, self.n_steps_output, self.n_fields, height, width)
        return out.permute(0, 1, 3, 4, 2)

    @torch.no_grad()
    def rollout(
        self,
        seed: torch.Tensor,
        scalar_conds: Optional[torch.Tensor],
        rollout_steps: int,
        normalizer=None,
        normalized_input: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        if seed.ndim != 5:
            raise ValueError(
                f"Expected seed shape `(B, Ti, H, W, F)`, got {tuple(seed.shape)}."
            )
        if rollout_steps < 1:
            raise ValueError(f"`rollout_steps` must be >= 1, got {rollout_steps}.")

        batch_size, steps_in, height, width, n_fields = seed.shape
        if steps_in < self.n_steps_input:
            raise ValueError(
                f"Seed has {steps_in} frames, but this WNO expects at least "
                f"{self.n_steps_input}."
            )
        if n_fields != self.n_fields:
            raise ValueError(f"Expected {self.n_fields} fields, got {n_fields}.")
        if not normalized_input and normalizer is None:
            raise ValueError(
                "`normalizer` is required when `normalized_input=False` so the "
                "model can roll out from physical-space seeds."
            )

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        output_dtype = torch.float32

        seed = seed.to(device=device, dtype=dtype)
        if normalized_input:
            windows_norm = seed[:, -self.n_steps_input :].clone()
            windows_phys = None
        else:
            windows_phys = seed[:, -self.n_steps_input :].clone()
            windows_norm = None

        scalar_conds_checked = self._check_scalar_conds(
            scalar_conds,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        preds = torch.zeros(
            batch_size,
            rollout_steps,
            height,
            width,
            self.n_fields,
            device="cpu",
            dtype=output_dtype,
        )

        step_out = 0
        num_calls = math.ceil(rollout_steps / self.n_steps_output)

        while step_out < rollout_steps:
            if verbose:
                call_idx = step_out // self.n_steps_output + 1
                print(
                    f"  rollout call {call_idx:>4d}/{num_calls} "
                    f"(frames {step_out + 1}-{min(step_out + self.n_steps_output, rollout_steps)}/"
                    f"{rollout_steps})",
                    end="\r",
                )

            if windows_phys is not None:
                window_norm = normalizer.normalize_flattened(windows_phys, mode="variable")
            else:
                window_norm = windows_norm

            flat = window_norm.permute(0, 1, 4, 2, 3).reshape(
                batch_size, self.in_channels, height, width
            )
            pred_flat = self(flat, scalar_conds=scalar_conds_checked)
            pred_norm = pred_flat.reshape(
                batch_size, self.n_steps_output, self.n_fields, height, width
            ).permute(0, 1, 3, 4, 2)

            if normalizer is not None:
                pred_store = normalizer.denormalize_flattened(pred_norm, mode="variable")
            else:
                pred_store = pred_norm

            frames_to_store = min(self.n_steps_output, rollout_steps - step_out)
            preds[:, step_out : step_out + frames_to_store] = pred_store[
                :, :frames_to_store
            ].detach().float().cpu()
            step_out += frames_to_store

            if windows_phys is not None:
                windows_phys = torch.cat((windows_phys, pred_store), dim=1)[:, -self.n_steps_input :]
            else:
                windows_norm = torch.cat((windows_norm, pred_norm), dim=1)[:, -self.n_steps_input :]

        if verbose:
            print()

        return preds


__all__ = ["WaveConv2d", "WNO2d"]

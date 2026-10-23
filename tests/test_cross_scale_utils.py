"""
Smoke test: exercise cross_scale_utils against a mock model that mimics the
WaveletFlowModule / WaveletFlowUNet API surface used by the library.
"""
import sys
sys.path.insert(0, "/mnt/user-data/outputs")

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

import cross_scale_utils as csu

C, T, H, W = 3, 1, 64, 128
J = 3


class MockDWT(nn.Module):
    """Non-redundant Haar-like pyramid: J-1 detail-only scales + coarsest with LL."""

    def forward_transform(self, x):          # (B,C,T,H,W)
        out, cur = {}, x
        for i in range(J):
            B, Ci, Ti, h, w = cur.shape
            a = cur[..., 0::2, 0::2]
            b = cur[..., 0::2, 1::2]
            c = cur[..., 1::2, 0::2]
            d = cur[..., 1::2, 1::2]
            LL = (a + b + c + d) / 2
            LH = (a - b + c - d) / 2
            HL = (a + b - c - d) / 2
            HH = (a - b - c + d) / 2
            if i == J - 1:
                out[i] = torch.stack([LL, LH, HL, HH], dim=3)
            else:
                out[i] = torch.stack([LH, HL, HH], dim=3)
            cur = LL
        return out

    def inverse_transform(self, coeffs):
        cur = None
        for i in reversed(range(J)):
            c = coeffs[i]
            if i == J - 1:
                LL, LH, HL, HH = c[:, :, :, 0], c[:, :, :, 1], c[:, :, :, 2], c[:, :, :, 3]
            else:
                LL = cur
                LH, HL, HH = c[:, :, :, 0], c[:, :, :, 1], c[:, :, :, 2]
            a = (LL + LH + HL + HH) / 2
            b = (LL - LH + HL - HH) / 2
            cc = (LL + LH - HL - HH) / 2
            d = (LL - LH - HL + HH) / 2
            B, Ci, Ti, h, w = LL.shape
            up = torch.zeros(B, Ci, Ti, h * 2, w * 2, dtype=LL.dtype, device=LL.device)
            up[..., 0::2, 0::2] = a
            up[..., 0::2, 1::2] = b
            up[..., 1::2, 0::2] = cc
            up[..., 1::2, 1::2] = d
            cur = up
        return cur


class MockNorm:
    def normalize_flattened(self, x, mode=None):
        return x
    def denormalize_flattened(self, x, mode=None):
        return x


class MockNet(nn.Module):
    """Genuinely couples scales: every head sees a pooled summary of all inputs."""

    def __init__(self, chans):
        super().__init__()
        self.chans = chans
        self.enc = nn.ModuleDict(
            {str(i): nn.Conv2d(2 * c, 8, 3, padding=1) for i, c in chans.items()}
        )
        self.head = nn.ModuleDict(
            {str(i): nn.Conv2d(8 + 8, c, 3, padding=1) for i, c in chans.items()}
        )

    def forward(self, curr, x_tau, t, scalar_conds=None, context=None):
        feats = {i: self.enc[str(i)](torch.cat([curr[i], x_tau[i]], 1))
                 for i in curr}
        glob = sum(f.mean(dim=(-2, -1)) for f in feats.values())      # cross-scale mix
        out = {}
        for i in feats:
            g = glob[:, :, None, None].expand(-1, -1, *feats[i].shape[-2:])
            out[i] = self.head[str(i)](torch.cat([feats[i], g], 1))
        return out


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_wavelet = True
        self.n_scales = J
        self.in_channels = C
        self.n_steps_output = T
        self.data_shape = (H, W)
        self.wavelet_decomposer = MockDWT()
        self.normalizer = MockNorm()
        probe = self.wavelet_decomposer.forward_transform(
            torch.zeros(1, C, T, H, W)
        )
        self.scale_shapes = {
            i: {"H": c.shape[-2], "W": c.shape[-1], "n_coeff": c.shape[3]}
            for i, c in probe.items()
        }
        chans = {i: C * T * s["n_coeff"] for i, s in self.scale_shapes.items()}
        self.net = MockNet(chans)

    def _to_coeff_dict(self, x):
        raw = self.wavelet_decomposer.forward_transform(x)
        return {i: rearrange(v, "B C T K H W -> B (C T K) H W")
                for i, v in raw.items()}

    def _to_coeff_dict_x0(self, x0):
        return self._to_coeff_dict(x0.expand(-1, -1, self.n_steps_output, -1, -1))


print("=" * 62)
model = MockModel().eval()

# 1 ── representation audit
audit = csu.check_representation(model)
assert audit["reconstruction_error"] < 1e-4, audit
assert abs(audit["redundancy"] - 1.0) < 1e-9, audit
print("\n[ok] check_representation\n")

# 2 ── probe batch
x_full = torch.randn(40, H, W, C)
label = torch.randn(2)
batch = csu.build_probe_batch(model, x_full, label, n_windows=4, n_steps_input=4)
print(batch.summary())
assert batch.B == 4 and batch.n_scales == J
for i in range(J):
    assert batch.c0[i].shape == batch.c1[i].shape
print("[ok] build_probe_batch\n")

# 3 ── sensitivity
S, floor = csu.sensitivity_matrix(model, batch, tau=0.5, delta=1e-2, n_probe=2)
assert S.shape == (J, J)
assert np.all(np.isfinite(S))
off = (S.sum() - np.trace(S)) / S.sum()
print(f"single tau: off-diag = {off:.1%}, floor = {floor:.1e}")
assert off > 1e-3, "mock model IS coupled; probe failed to detect it"
print("[ok] sensitivity_matrix\n")

res = csu.sensitivity_sweep(model, batch, taus=(0.2, 0.5, 0.8), n_probe=2, seed=0)
assert res["mean"].shape == (J, J)
print("[ok] sensitivity_sweep\n")

import matplotlib
matplotlib.use("Agg")
fig = csu.plot_sensitivity(res, savepath="/tmp/sens.pdf")
print("[ok] plot_sensitivity\n")

# 3b ── negative control: a genuinely factorized net must give ZERO off-diagonal
class FactorizedNet(nn.Module):
    def __init__(self, chans):
        super().__init__()
        self.f = nn.ModuleDict(
            {str(i): nn.Conv2d(2 * c, c, 3, padding=1) for i, c in chans.items()}
        )
    def forward(self, curr, x_tau, t, scalar_conds=None, context=None):
        return {i: self.f[str(i)](torch.cat([curr[i], x_tau[i]], 1)) for i in curr}

chans = {i: C * T * s["n_coeff"] for i, s in model.scale_shapes.items()}
model_f = MockModel().eval()
model_f.net = FactorizedNet(chans).eval()
Sf, floor_f = csu.sensitivity_matrix(model_f, batch, tau=0.5, n_probe=2)
off_f = (Sf.sum() - np.trace(Sf)) / Sf.sum()
print(f"factorized control: off-diag = {off_f:.3e} (must be ~0)")
assert off_f < 1e-9, f"probe reports coupling for an independent model! {Sf}"
print("[ok] negative control — probe does not produce false positives\n")

# 4 ── wavelet stats
fields = torch.randn(1, 12, H, W, C)
st = csu.wavelet_stats(model, fields)
print("gaussian field kurtosis:", {k: round(v, 2) for k, v in st["kurtosis"].items()})
assert all(2.0 < v < 4.5 for v in st["kurtosis"].values()), st
assert all(abs(v) < 0.35 for v in st["parent_child"].values()), st

heavy = torch.randn(1, 12, H, W, C)
mask = (torch.rand(1, 12, H, W, 1) < 0.03).float()
heavy = heavy + mask * torch.randn(1, 12, H, W, C) * 12   # spatially-localised bursts
st_h = csu.wavelet_stats(model, heavy)
print("bursty field  kurtosis:", {k: round(v, 2) for k, v in st_h["kurtosis"].items()})
print("bursty parent-child   :", {k: round(v, 3) for k, v in st_h["parent_child"].items()})
assert max(st_h["kurtosis"].values()) > max(st["kurtosis"].values()), "kurtosis insensitive"
assert max(st_h["parent_child"].values()) > max(st["parent_child"].values()), \
    "parent-child insensitive"
print("[ok] wavelet_stats discriminates gaussian vs intermittent\n")

fig = csu.plot_wavelet_stats({"gaussian": st, "bursty": st_h},
                             savepath="/tmp/stats.pdf")
print("[ok] plot_wavelet_stats\n")

# 5 ── clamp report
rep = csu.clamp_report(torch.randn(2, 5, H, W, C) * 2.0, clamp_sigma=5.0)
print("clamp_report:", {k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in rep.items()})
print("[ok] clamp_report\n")

print("=" * 62)
print("ALL SMOKE TESTS PASSED")

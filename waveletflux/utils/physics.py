"""
Physics-domain metrics for WaveletFlow evaluation.

Companion to `eval.py`. Nothing in `eval.py` is modified or imported; this
module is additive and mirrors its conventions (same argument names, same
long-format DataFrame contract, same ensemble handling).

────────────────────────────────────────────────────────────────────────────
TURNING METRICS ON AND OFF
────────────────────────────────────────────────────────────────────────────
Every metric lives in a registry and is selected by name. You never comment
out code -- you pass a list:

    build_physics_global_df(..., metrics=["kinetic_energy", "enstrophy",
                                          "divergence", "nusselt"])

`physics.list_metrics()` prints the registry with its cost tag. The default
lists are in `RECOMMENDED[<dataset>]` and are deliberately short.

COST (measured at 512x128 with 16 ensemble members, per frame):

    kinetic_energy        7.6 ms      cheap
    vorticity (shared)   22.0 ms      cheap   -- computed once, reused
    enstrophy             3.1 ms      cheap
    palinstrophy         14.6 ms      cheap
    divergence           71.2 ms      cheap
    nusselt               2.0 ms      cheap
    gradsq (per field)   16.2 ms      cheap
    ------------------------------------------------
    ALL of the above    136.5 ms
    vorticity_pdf (W1)  589.7 ms      EXPENSIVE  -- 4x everything else
    field_pdf     (W1)  ~590 ms       EXPENSIVE  -- per field!

Over a full Rayleigh-Benard run (9 trajectories x 196 steps) that is 4
minutes for everything cheap versus 21 minutes once the Wasserstein terms
are on. The `*_pdf` metrics are the only ones that cost anything; they are
OFF by default. Turn one on for a single final table if you want the
invariant-measure number, and leave it off during development.

────────────────────────────────────────────────────────────────────────────
THREE LAYERS
────────────────────────────────────────────────────────────────────────────
Layer 1  `derive_fields`   -- build physical channels (vorticity, KE, ...) and
         append them to the field axis so the EXISTING build_stats_df /
         build_psd_df evaluate them unchanged. The PSD of the `vorticity`
         channel is the enstrophy spectrum k^2 E(k).
Layer 2  `build_physics_field_df`  -- per-(model, traj, field, step) metrics.
Layer 3  `build_physics_global_df` -- per-(model, traj, step) metrics that
         couple channels (enstrophy, Nu, continuity residual, ...).

────────────────────────────────────────────────────────────────────────────
ENSEMBLE CONVENTION
────────────────────────────────────────────────────────────────────────────
Every quantity is evaluated per ensemble member and then reduced; the
ensemble-mean field is never passed to a nonlinear functional. Each quantity
Q is reported four ways:

    <q>_true   Q(x_true)
    <q>_pred   mean_m Q(x_m)
    <q>_bias   |mean_m Q(x_m) - Q(x_true)| / |Q(x_true)|   <- the table number
    <q>_mae    mean_m |Q(x_m) - Q(x_true)| / |Q(x_true)|

────────────────────────────────────────────────────────────────────────────
GRID
────────────────────────────────────────────────────────────────────────────
`rayleigh_benard` is sampled at Chebyshev nodes in the vertical. Uniform-
spacing differencing gives ~75x relative error on d/dy for a resolved mode,
which corrupts vorticity, divergence, Nu and every dissipation rate. Prefer
`Grid.from_coords(...)`; the presets are an analytic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

import ot
import pandas as pd
import torch

_EPS = 1e-12
_GAMMA_TRL = 5.0 / 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Grid
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Grid:
    """
    2-D grid. A frame is (..., Nx, Ny) with x_axis=-2, y_axis=-1, matching
    The Well's (time, x, y, field) layout -- for rayleigh_benard the stored
    resolution is 512 x 128, so Nx=512, Ny=128. If your loader gives (Ny, Nx),
    pass x_axis=-1, y_axis=-2.
    """
    x: torch.Tensor
    y: torch.Tensor
    periodic_x: bool = True
    periodic_y: bool = False
    period_x: float | None = None
    period_y: float | None = None
    x_axis: int = -2
    y_axis: int = -1

    @staticmethod
    def from_coords(x, y, periodic_x=True, periodic_y=False,
                    period_x=None, period_y=None, **kw) -> "Grid":
        x = torch.as_tensor(x, dtype=torch.float64).flatten()
        y = torch.as_tensor(y, dtype=torch.float64).flatten()
        if periodic_x and period_x is None:
            period_x = float(x[-1] - x[0] + (x[1] - x[0]))
        if periodic_y and period_y is None:
            period_y = float(y[-1] - y[0] + (y[1] - y[0]))
        return Grid(x, y, periodic_x, periodic_y, period_x, period_y, **kw)

    @staticmethod
    def uniform(Lx, Nx, Ly, Ny, x0=0.0, y0=0.0,
                periodic_x=True, periodic_y=False) -> "Grid":
        x = (x0 + torch.arange(Nx, dtype=torch.float64) * (Lx / Nx) if periodic_x
             else torch.linspace(x0, x0 + Lx, Nx, dtype=torch.float64))
        y = (y0 + torch.arange(Ny, dtype=torch.float64) * (Ly / Ny) if periodic_y
             else torch.linspace(y0, y0 + Ly, Ny, dtype=torch.float64))
        return Grid(x, y, periodic_x, periodic_y,
                    Lx if periodic_x else None, Ly if periodic_y else None)

    @staticmethod
    def chebyshev_y(Lx, Nx, Ly, Ny, x0=0.0, y0=0.0) -> "Grid":
        x = x0 + torch.arange(Nx, dtype=torch.float64) * (Lx / Nx)
        j = torch.arange(Ny, dtype=torch.float64)
        y = y0 + Ly * 0.5 * (1.0 - torch.cos(torch.pi * j / (Ny - 1)))
        return Grid(x, y, True, False, Lx, None)

    def _w1d(self, nodes, periodic, period):
        if periodic:
            return torch.full_like(nodes, float(period) / len(nodes))
        w = torch.zeros_like(nodes)
        w[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
        w[0] = 0.5 * (nodes[1] - nodes[0])
        w[-1] = 0.5 * (nodes[-1] - nodes[-2])
        return w

    def weights(self, device, dtype) -> torch.Tensor:
        wx = self._w1d(self.x, self.periodic_x, self.period_x)
        wy = self._w1d(self.y, self.periodic_y, self.period_y)
        w = wx[:, None] * wy[None, :]
        return (w / w.sum()).to(device=device, dtype=dtype)

    @property
    def area(self) -> float:
        wx = self._w1d(self.x, self.periodic_x, self.period_x)
        wy = self._w1d(self.y, self.periodic_y, self.period_y)
        return float(wx.sum() * wy.sum())

    def avg(self, f): return (f * self.weights(f.device, f.dtype)).sum(dim=(-2, -1))
    def integral(self, f): return self.avg(f) * self.area

    def d_dx(self, f):
        return _derivative(f, self.x, self.x_axis, self.periodic_x, self.period_x)

    def d_dy(self, f):
        return _derivative(f, self.y, self.y_axis, self.periodic_y, self.period_y)

    def grad(self, f): return self.d_dx(f), self.d_dy(f)


def _derivative(f, nodes, axis, periodic, period) -> torch.Tensor:
    """2nd-order central difference, exact for non-uniform node spacing."""
    nodes = nodes.to(device=f.device, dtype=f.dtype)
    f = f.movedim(axis, -1)
    if periodic:
        h = float(period) / nodes.numel()
        out = (torch.roll(f, -1, dims=-1) - torch.roll(f, 1, dims=-1)) / (2.0 * h)
        return out.movedim(-1, axis)
    hm = nodes[1:-1] - nodes[:-2]
    hp = nodes[2:] - nodes[1:-1]
    d_int = ((-hp / (hm * (hm + hp))) * f[..., :-2]
             + ((hp - hm) / (hm * hp)) * f[..., 1:-1]
             + (hm / (hp * (hm + hp))) * f[..., 2:])
    d0 = ((f[..., 1] - f[..., 0]) / (nodes[1] - nodes[0])).unsqueeze(-1)
    dN = ((f[..., -1] - f[..., -2]) / (nodes[-1] - nodes[-2])).unsqueeze(-1)
    return torch.cat([d0, d_int, dN], dim=-1).movedim(-1, axis)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset specification
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DatasetSpec:
    """
    Per-PDE configuration: which raw channels play which physical role, which
    fields must remain positive, which carry a diffusivity.

    `roles` maps a canonical role -> the field name in your `field_names`.
    OVERRIDE with `spec.with_roles(ux="...", uy="...")` if they do not match.
    """
    name: str
    roles: dict[str, str]
    grid: Grid
    dt: float
    positive_fields: tuple[str, ...] = ()
    diffusivity: dict[str, str] | None = None
    derived: tuple[str, ...] = ("vorticity", "kinetic_energy")
    incompressible: bool = True

    def with_grid(self, grid): return replace(self, grid=grid)
    def with_roles(self, **r): return replace(self, roles={**self.roles, **r})

    def resolve(self, field_names: Sequence[str]) -> dict[str, int]:
        lut = {n: i for i, n in enumerate(field_names)}
        return {r: lut[n] for r, n in self.roles.items() if n in lut}


DATASET_SPECS: dict[str, DatasetSpec] = {
    "active_matter": DatasetSpec(
        name="active_matter",
        roles={"ux": "velocity_x", "uy": "velocity_y", "c":"concentration",
           "Dxx":"D_xx", "Dxy":"D_xy", "Dyy":"D_yy",
           "Exx":"E_xx", "Exy":"E_xy", "Eyy":"E_yy"},
        grid=Grid.uniform(10.0, 256, 10.0, 256, periodic_x=True, periodic_y=True),
        dt=0.25,
        positive_fields=("concentration",),
        diffusivity={},
        derived=("vorticity", "kinetic_energy"),
        incompressible=True,
    ),
    "rayleigh_benard": DatasetSpec(
        name="rayleigh_benard",
        roles={"ux": "velocity_x", "uy": "velocity_y",
               "b": "buoyancy", "p": "pressure"},
        grid=Grid.chebyshev_y(4.0, 512, 1.0, 128),
        dt=0.25,
        positive_fields=(),
        diffusivity={"buoyancy": "kappa", "velocity_x": "nu", "velocity_y": "nu"},
        derived=("vorticity", "kinetic_energy"),
        incompressible=True,
    ),
    "turbulent_radiative_layer_2D": DatasetSpec(
        name="turbulent_radiative_layer_2D",
        roles={"ux": "velocity_x", "uy": "velocity_y",
               "rho": "density", "p": "pressure"},
        grid=Grid.uniform(1.0, 384, 3.0, 128, x0=-0.5, y0=-1.0,
                          periodic_x=True, periodic_y=False),
        dt=1.597033,
        positive_fields=("density", "pressure"),
        diffusivity={},
        derived=("vorticity", "kinetic_energy"),
        incompressible=False,
    ),
}


def spec_for(dataset_name: str) -> DatasetSpec:
    """Tolerant lookup: handles the *_uniform and *_3D name variants."""
    if dataset_name in DATASET_SPECS:
        return DATASET_SPECS[dataset_name]
    if dataset_name.startswith("rayleigh_benard"):
        return DATASET_SPECS["rayleigh_benard"]
    if dataset_name.startswith("turbulent_radiative_layer"):
        return DATASET_SPECS["turbulent_radiative_layer_2D"]
    raise KeyError(f"no DatasetSpec for '{dataset_name}'")


# ══════════════════════════════════════════════════════════════════════════════
# Shared reduction helpers
# ══════════════════════════════════════════════════════════════════════════════

def _reduce(q_true: float, q_pred: torch.Tensor, name: str, eps=_EPS) -> dict:
    q_pred = q_pred.reshape(-1).double()
    denom = abs(q_true) + eps
    return {f"{name}_true": float(q_true),
            f"{name}_pred": float(q_pred.mean()),
            f"{name}_bias": float((q_pred.mean() - q_true).abs() / denom),
            f"{name}_mae":  float((q_pred - q_true).abs().mean() / denom)}


def _w1(a, b) -> float:
    return float(ot.lp.wasserstein_1d(a.flatten().float().cpu(),
                                      b.flatten().float().cpu(), p=1.0))


def _w1_weighted(va, wa, vb, wb, bins) -> float:
    def _cdf(v, w):
        h = torch.histogram(v.flatten().float().cpu(), bins=bins.cpu(),
                            weight=w.flatten().float().cpu()).hist.clamp(min=0)
        s = h.sum()
        return (h / s).cumsum(0) if s > 0 else h
    return float(((_cdf(va, wa) - _cdf(vb, wb)).abs()
                  * (bins[1:] - bins[:-1]).cpu()).sum())


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation context -- shared, lazily-cached intermediates
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Ctx:
    """
    Holds one (truth frame, ensemble of predicted frames) pair plus everything
    derived from it. Intermediates such as the vorticity are computed once on
    first access and reused by every metric that needs them, so enabling both
    `enstrophy` and `palinstrophy` costs one vorticity, not two.
    """
    u: torch.Tensor           # (Nx, Ny, C)
    v: torch.Tensor           # (N_ENS, Nx, Ny, C)
    grid: Grid
    idx: Mapping[str, int]
    params: Mapping
    eps: float = _EPS
    _c: dict = field(default_factory=dict)

    def t(self, role): return self.u[..., self.idx[role]]
    def p(self, role): return self.v[..., self.idx[role]]
    def has(self, *roles): return all(r in self.idx for r in roles)

    def _memo(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    @property
    def w_t(self):
        return self._memo("w_t", lambda: self.grid.d_dx(self.t("uy"))
                                         - self.grid.d_dy(self.t("ux")))

    @property
    def w_p(self):
        return self._memo("w_p", lambda: self.grid.d_dx(self.p("uy"))
                                         - self.grid.d_dy(self.p("ux")))

    @property
    def rho_t(self): return self.t("rho") if self.has("rho") else None

    @property
    def rho_p(self): return self.p("rho") if self.has("rho") else None


@dataclass(frozen=True)
class Metric:
    fn: Callable[[Ctx], dict]
    roles: tuple[str, ...]
    cost: str          # "cheap" | "expensive"
    doc: str


GLOBAL_METRICS: dict[str, Metric] = {}


def _global(name, roles=("ux", "uy"), cost="cheap", doc=""):
    def deco(fn):
        GLOBAL_METRICS[name] = Metric(fn, roles, cost, doc or (fn.__doc__ or "").strip())
        return fn
    return deco


# ── Universal ─────────────────────────────────────────────────────────────────

@_global("kinetic_energy")
def _g_ke(c: Ctx) -> dict:
    """<0.5 rho |u|^2>. rho == 1 for the incompressible systems."""
    def ke(ax, ay, r):
        return c.grid.avg(0.5 * (ax ** 2 + ay ** 2) * (1.0 if r is None else r))
    return _reduce(float(ke(c.t("ux"), c.t("uy"), c.rho_t)),
                   ke(c.p("ux"), c.p("uy"), c.rho_p), "ke", c.eps)


@_global("urms")
def _g_urms(c: Ctx) -> dict:
    """sqrt(<|u|^2>), the rms velocity."""
    return _reduce(float(c.grid.avg(c.t("ux") ** 2 + c.t("uy") ** 2).sqrt()),
                   c.grid.avg(c.p("ux") ** 2 + c.p("uy") ** 2).sqrt(),
                   "urms", c.eps)


@_global("enstrophy")
def _g_enstrophy(c: Ctx) -> dict:
    """<0.5 omega^2>. Weighted towards small scales, so it exposes the
    over-smoothing that VRMSE hides."""
    return _reduce(float(0.5 * c.grid.avg(c.w_t ** 2)),
                   0.5 * c.grid.avg(c.w_p ** 2), "enstrophy", c.eps)


@_global("palinstrophy")
def _g_palinstrophy(c: Ctx) -> dict:
    """<0.5 |grad omega|^2>, one derivative order finer than enstrophy."""
    def pal(w): return 0.5 * c.grid.avg(c.grid.d_dx(w) ** 2 + c.grid.d_dy(w) ** 2)
    return _reduce(float(pal(c.w_t)), pal(c.w_p), "palinstrophy", c.eps)


@_global("divergence")
def _g_divergence(c: Ctx) -> dict:
    """||div u|| / ||grad u||_F, the dimensionless incompressibility residual.
    Ground truth sits at the solver's own numerical floor, so read the RATIO."""
    def dn(ax, ay):
        d = c.grid.d_dx(ax) + c.grid.d_dy(ay)
        g = (c.grid.d_dx(ax) ** 2 + c.grid.d_dy(ax) ** 2
             + c.grid.d_dx(ay) ** 2 + c.grid.d_dy(ay) ** 2)
        return (c.grid.avg(d ** 2) / (c.grid.avg(g) + _EPS)).sqrt()
    dt_ = float(dn(c.t("ux"), c.t("uy")))
    dp_ = dn(c.p("ux"), c.p("uy"))
    return {"divnorm_true": dt_, "divnorm_pred": float(dp_.mean()),
            "divnorm_ratio": float(dp_.mean() / (dt_ + _EPS))}


@_global("vorticity_pdf", cost="expensive")
def _g_w1_vort(c: Ctx) -> dict:
    """EXPENSIVE (~590 ms/frame at 512x128, 16 members). W1 between the true
    and predicted vorticity PDFs -- the invariant-measure fidelity number."""
    per = [_w1(c.w_t, c.w_p[m]) for m in range(c.w_p.shape[0])]
    return {"w1_vort": float(torch.tensor(per).mean()),
            "w1_vort_pooled": _w1(c.w_t, c.w_p)}


# ── Rayleigh-Benard ───────────────────────────────────────────────────────────

def _rb_constants(c: Ctx):
    kappa, nu = c.params.get("kappa"), c.params.get("nu")
    if kappa is None and {"Ra", "Pr"} <= set(c.params):
        kappa = (c.params["Ra"] * c.params["Pr"]) ** -0.5
        nu = (c.params["Ra"] / c.params["Pr"]) ** -0.5
    return kappa, nu


@_global("nusselt", roles=("ux", "uy", "b"))
def _g_nusselt(c: Ctx) -> dict:
    """Nu = 1 + <u_y b> L_y / (kappa delta_b). The canonical RB diagnostic."""
    kappa, _ = _rb_constants(c)
    if kappa is None:
        return {}
    Ly = c.params.get("Ly", float(c.grid.y[-1] - c.grid.y[0]))
    scale = kappa * c.params.get("delta_b", 1.0) / Ly
    return _reduce(float(1.0 + c.grid.avg(c.t("uy") * c.t("b")) / scale),
                   1.0 + c.grid.avg(c.p("uy") * c.p("b")) / scale,
                   "nusselt", c.eps)


@_global("thermal_dissipation", roles=("b",))
def _g_eps_b(c: Ctx) -> dict:
    """eps_b = kappa <|grad b|^2>, the thermal dissipation rate."""
    kappa, _ = _rb_constants(c)
    if kappa is None:
        return {}
    def gb(b): return kappa * c.grid.avg(c.grid.d_dx(b) ** 2 + c.grid.d_dy(b) ** 2)
    return _reduce(float(gb(c.t("b"))), gb(c.p("b")), "eps_b", c.eps)


@_global("viscous_dissipation", roles=("ux", "uy", "b"))
def _g_eps_u(c: Ctx) -> dict:
    """eps_u = nu <|grad u|^2>, the viscous dissipation rate."""
    _, nu = _rb_constants(c)
    if nu is None:
        return {}
    def ev(ax, ay):
        return nu * c.grid.avg(c.grid.d_dx(ax) ** 2 + c.grid.d_dy(ax) ** 2
                               + c.grid.d_dx(ay) ** 2 + c.grid.d_dy(ay) ** 2)
    return _reduce(float(ev(c.t("ux"), c.t("uy"))),
                   ev(c.p("ux"), c.p("uy")), "eps_u", c.eps)


@_global("reynolds", roles=("ux", "uy", "b"))
def _g_reynolds(c: Ctx) -> dict:
    """Re = U_rms L / nu."""
    _, nu = _rb_constants(c)
    if nu is None:
        return {}
    L = float(c.grid.y[-1] - c.grid.y[0])
    return _reduce(float(c.grid.avg(c.t("ux") ** 2 + c.t("uy") ** 2).sqrt() * L / nu),
                   c.grid.avg(c.p("ux") ** 2 + c.p("uy") ** 2).sqrt() * L / nu,
                   "reynolds", c.eps)


# ── Turbulent radiative layer ─────────────────────────────────────────────────

@_global("massflux", roles=("rho", "uy"))
def _g_massflux(c: Ctx) -> dict:
    """<rho u_y>, the hot-to-cold mass transfer rate -- the quantity the
    domain expert names as the evaluation criterion for this dataset."""
    return _reduce(float(c.grid.avg(c.rho_t * c.t("uy"))),
                   c.grid.avg(c.rho_p * c.p("uy")), "massflux", c.eps)


@_global("cooling", roles=("p",))
def _g_cooling(c: Ctx) -> dict:
    """<P / ((gamma-1) t_cool)>, the total radiative sink."""
    tc = c.params.get("t_cool", 1.0)
    return _reduce(float(c.grid.avg(c.t("p")) / (_GAMMA_TRL - 1.0) / tc),
                   c.grid.avg(c.p("p")) / (_GAMMA_TRL - 1.0) / tc, "cooling", c.eps)


@_global("cold_fraction", roles=("rho", "p"))
def _g_cold_frac(c: Ctx) -> dict:
    """Mass fraction with T = P/rho below params['t_cold']."""
    thr = c.params.get("t_cold")
    if thr is None:
        return {}
    def cf(pp, rr):
        T = pp / rr.clamp(min=_EPS)
        return c.grid.avg(rr * (T < thr).double()) / c.grid.avg(rr)
    return _reduce(float(cf(c.t("p"), c.rho_t)), cf(c.p("p"), c.rho_p),
                   "cold_frac", c.eps)


@_global("temperature_pdf", roles=("rho", "p"), cost="expensive")
def _g_w1_logT(c: Ctx) -> dict:
    """EXPENSIVE. W1 between mass-weighted log10(T) distributions -- the
    "mass per temperature bin" diagnostic for TRL."""
    lT_t = torch.log10((c.t("p") / c.rho_t.clamp(min=_EPS)).clamp(min=_EPS))
    lT_p = torch.log10((c.p("p") / c.rho_p.clamp(min=_EPS)).clamp(min=_EPS))
    lo = float(torch.quantile(lT_t.flatten(), 0.001))
    hi = float(torch.quantile(lT_t.flatten(), 0.999))
    bins = torch.linspace(lo - 0.2, hi + 0.2, 129, dtype=torch.float32)
    per = [_w1_weighted(lT_t, c.rho_t, lT_p[m], c.rho_p[m], bins)
           for m in range(lT_p.shape[0])]
    return {"w1_logT": float(torch.tensor(per).mean())}


# ── Active matter ─────────────────────────────────────────────────────────────

def _nematic_field(t, idx):
    dxx, dxy = t[..., idx["Dxx"]], t[..., idx["Dxy"]]
    dyy = t[..., idx["Dyy"]] if "Dyy" in idx else -dxx
    q = 0.5 * (dxx - dyy)
    return 2.0 * torch.sqrt(q ** 2 + dxy ** 2)


def _active_power_field(t, idx):
    dxx, dxy = t[..., idx["Dxx"]], t[..., idx["Dxy"]]
    exx, exy = t[..., idx["Exx"]], t[..., idx["Exy"]]
    dyy = t[..., idx["Dyy"]] if "Dyy" in idx else -dxx
    eyy = t[..., idx["Eyy"]] if "Eyy" in idx else -exx
    return dxx * exx + 2.0 * dxy * exy + dyy * eyy


@_global("nematic", roles=("Dxx", "Dxy"))
def _g_nematic(c: Ctx) -> dict:
    """Domain-mean scalar order parameter S = 2 sqrt(Qxx^2 + Qxy^2)."""
    return _reduce(float(c.grid.avg(_nematic_field(c.u, c.idx))),
                   c.grid.avg(_nematic_field(c.v, c.idx)), "nematic", c.eps)


@_global("active_power", roles=("Dxx", "Dxy", "Exx", "Exy"))
def _g_activepow(c: Ctx) -> dict:
    """<D : E>, the active energy injection."""
    return _reduce(float(c.grid.avg(_active_power_field(c.u, c.idx))),
                   c.grid.avg(_active_power_field(c.v, c.idx)), "activepow", c.eps)


@_global("concentration_variance", roles=("c",))
def _g_conc_var(c: Ctx) -> dict:
    """<(c - <c>)^2>, concentration clustering."""
    def cv(t, is_pred):
        f = t[..., c.idx["c"]]
        return c.grid.avg((f - c.grid.avg(f)[..., None, None]) ** 2)
    return _reduce(float(cv(c.u, False)), cv(c.v, True), "conc_var", c.eps)


# ══════════════════════════════════════════════════════════════════════════════
# Per-field metric registry
# ══════════════════════════════════════════════════════════════════════════════

FIELD_METRICS: dict[str, Metric] = {}


def _fieldm(name, cost="cheap"):
    def deco(fn):
        FIELD_METRICS[name] = Metric(fn, (), cost, (fn.__doc__ or "").strip())
        return fn
    return deco


@dataclass
class FieldCtx:
    u: torch.Tensor                 # (Nx, Ny)
    v: torch.Tensor                 # (N_ENS, Nx, Ny)
    grid: Grid
    diffusivity: float | None = None
    positive: bool = False
    eps: float = _EPS
    _c: dict = field(default_factory=dict)

    def _memo(self, k, fn):
        if k not in self._c:
            self._c[k] = fn()
        return self._c[k]

    @property
    def gradsq_t(self):
        return self._memo("gt", lambda: float(self.grid.avg(
            self.grid.d_dx(self.u) ** 2 + self.grid.d_dy(self.u) ** 2)))

    @property
    def gradsq_p(self):
        return self._memo("gp", lambda: self.grid.avg(
            self.grid.d_dx(self.v) ** 2 + self.grid.d_dy(self.v) ** 2))


@_fieldm("gradsq")
def _f_gradsq(c: FieldCtx) -> dict:
    """<|grad f|^2>. The sharpest per-field detector of over-smoothing: a
    mean-regression emulator can hold its VRMSE while shedding a large
    fraction of its gradient energy."""
    return _reduce(c.gradsq_t, c.gradsq_p, "gradsq", c.eps)


@_fieldm("dissipation")
def _f_dissipation(c: FieldCtx) -> dict:
    """kappa_f <|grad f|^2>, with the physical prefactor from the spec."""
    if c.diffusivity is None:
        return {}
    return _reduce(c.diffusivity * c.gradsq_t,
                   c.diffusivity * c.gradsq_p, "diss", c.eps)


@_fieldm("mass")
def _f_mass(c: FieldCtx) -> dict:
    """Grid-weighted domain integral of f. Uses proper quadrature weights --
    an unweighted sum is wrong on the Chebyshev vertical grid of RB."""
    return _reduce(float(c.grid.integral(c.u)), c.grid.integral(c.v), "mass", c.eps)


@_fieldm("extremes")
def _f_extremes(c: FieldCtx) -> dict:
    """1st / 99th percentile and peak amplitude preservation."""
    uf, vf = c.u.flatten(), c.v.reshape(c.v.shape[0], -1)
    q = torch.tensor([0.01, 0.99], dtype=uf.dtype)
    qt, qp = torch.quantile(uf, q), torch.quantile(vf, q, dim=1)
    out = {"p01_bias": float((qp[0].mean() - qt[0]).abs() / (qt[0].abs() + c.eps)),
           "p99_bias": float((qp[1].mean() - qt[1]).abs() / (qt[1].abs() + c.eps))}
    return out | _reduce(float(uf.abs().max()), vf.abs().amax(dim=1), "absmax", c.eps)


@_fieldm("positivity")
def _f_positivity(c: FieldCtx) -> dict:
    """Domain fraction with f < 0, for fields that must stay positive.
    Density, pressure and concentration are strictly positive in the raw data
    and a generative emulator has no structural reason to respect that."""
    if not c.positive:
        return {}
    w = c.grid.weights(c.u.device, c.u.dtype)
    return {"neg_frac_true": float(((c.u < 0).double() * w).sum()),
            "neg_frac_pred": float(((c.v < 0).double() * w).sum(dim=(-2, -1)).mean())}


@_fieldm("field_pdf", cost="expensive")
def _f_pdf(c: FieldCtx) -> dict:
    """EXPENSIVE (~590 ms/frame PER FIELD). W1 between the true and predicted
    marginal PDFs of this field."""
    per = [_w1(c.u, c.v[m]) for m in range(c.v.shape[0])]
    return {"w1_pdf": float(torch.tensor(per).mean()),
            "w1_pdf_pooled": _w1(c.u, c.v)}


# ══════════════════════════════════════════════════════════════════════════════
# Recommended defaults -- three cheap metrics per dataset
# ══════════════════════════════════════════════════════════════════════════════

RECOMMENDED: dict[str, dict[str, list[str]]] = {
    # Enstrophy (small-scale fidelity) + a conservation residual + the
    # dataset's own flux diagnostic. Everything else is off.
    "rayleigh_benard": {
        "global": ["kinetic_energy", "enstrophy", "divergence", "nusselt"],
        "field":  ["gradsq"],
        "continuity": False,          # divergence already covers it
    },
    "turbulent_radiative_layer_2D": {
        "global": ["kinetic_energy", "enstrophy", "massflux"],
        "field":  ["gradsq", "positivity"],
        "continuity": True,           # compressible: needs two frames
    },
    "active_matter": {
        "global": ["kinetic_energy", "enstrophy", "divergence", "nematic"],
        "field":  ["gradsq"],
        "continuity": False,
    },
}


def recommended_for(dataset_name: str) -> dict:
    if dataset_name in RECOMMENDED:
        return RECOMMENDED[dataset_name]
    if dataset_name.startswith("rayleigh_benard"):
        return RECOMMENDED["rayleigh_benard"]
    if dataset_name.startswith("turbulent_radiative_layer"):
        return RECOMMENDED["turbulent_radiative_layer_2D"]
    raise KeyError(f"no recommended metric set for '{dataset_name}'")


def list_metrics() -> None:
    """Print the registry so you can see what is available and what it costs."""
    for title, reg in (("GLOBAL (build_physics_global_df)", GLOBAL_METRICS),
                       ("PER-FIELD (build_physics_field_df)", FIELD_METRICS)):
        print(f"\n{title}")
        print("-" * 78)
        for name, m in reg.items():
            tag = "EXPENSIVE" if m.cost == "expensive" else "cheap"
            roles = f" roles={list(m.roles)}" if m.roles else ""
            print(f"  {name:<24s} [{tag:^9s}]{roles}")
            print(f"      {m.doc.splitlines()[0] if m.doc else ''}")
    print("\n  'continuity' is a separate flag on build_physics_global_df "
          "(needs two frames).\n")


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 -- derived physical fields
# ══════════════════════════════════════════════════════════════════════════════

DERIVED_BUILDERS: dict[str, Callable] = {}
DERIVED_REQUIRES: dict[str, tuple[str, ...]] = {}


def _derived(name, requires):
    def deco(fn):
        DERIVED_BUILDERS[name] = fn
        DERIVED_REQUIRES[name] = requires
        return fn
    return deco


@_derived("vorticity", ("ux", "uy"))
def _d_vorticity(t, idx, grid, params):
    """omega = d(uy)/dx - d(ux)/dy. Its PSD is the enstrophy spectrum."""
    return grid.d_dx(t[..., idx["uy"]]) - grid.d_dy(t[..., idx["ux"]])


@_derived("kinetic_energy", ("ux", "uy"))
def _d_ke(t, idx, grid, params):
    e = 0.5 * (t[..., idx["ux"]] ** 2 + t[..., idx["uy"]] ** 2)
    return e if "rho" not in idx else t[..., idx["rho"]] * e


@_derived("divergence", ("ux", "uy"))
def _d_div(t, idx, grid, params):
    return grid.d_dx(t[..., idx["ux"]]) + grid.d_dy(t[..., idx["uy"]])


@_derived("heat_flux", ("uy", "b"))
def _d_heat_flux(t, idx, grid, params):
    return t[..., idx["uy"]] * t[..., idx["b"]]


@_derived("temperature", ("p", "rho"))
def _d_temperature(t, idx, grid, params):
    return t[..., idx["p"]] / t[..., idx["rho"]].clamp(min=_EPS)


@_derived("mass_flux_y", ("rho", "uy"))
def _d_mass_flux_y(t, idx, grid, params):
    return t[..., idx["rho"]] * t[..., idx["uy"]]


@_derived("nematic_order", ("Dxx", "Dxy"))
def _d_nematic(t, idx, grid, params):
    return _nematic_field(t, idx)


def _flat_params(params, traj_idx):
    if not params:
        return {}
    if traj_idx in params and isinstance(params[traj_idx], Mapping):
        return dict(params[traj_idx])
    return dict(params)


def derive_fields(truth, predictions, field_names, spec,
                  derived=None, params=None, append=True):
    """
    Build derived physical channels and append them to the field axis, so the
    EXISTING build_stats_df / build_psd_df evaluate them with no change to
    eval.py.

    Call on DENORMALISED fields -- the vorticity of a z-scored velocity is not
    a vorticity.

    Returns (truth_out, predictions_out, field_names_out).
    """
    idx = spec.resolve(field_names)
    names = list(derived) if derived is not None else list(spec.derived)
    unknown = [n for n in names if n not in DERIVED_BUILDERS]
    if unknown:
        raise KeyError(f"unknown derived fields {unknown}; "
                       f"available: {sorted(DERIVED_BUILDERS)}")
    keep = [n for n in names if all(r in idx for r in DERIVED_REQUIRES[n])]
    dropped = [n for n in names if n not in keep]
    if dropped:
        print(f"[physics] skipping derived fields {dropped} (missing roles); "
              f"resolved: {sorted(idx)}")

    def _build(t):
        chans = []
        for n in keep:
            per = [DERIVED_BUILDERS[n](t[i:i + 1].double(), idx, spec.grid,
                                       _flat_params(params, i))
                   for i in range(t.shape[0])]
            chans.append(torch.cat(per, dim=0).to(t.dtype))
        st = torch.stack(chans, dim=-1)
        return torch.cat([t, st], dim=-1) if append else st

    names_out = (list(field_names) + keep) if append else keep
    return _build(truth), {k: _build(v) for k, v in predictions.items()}, names_out


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 -- per-field metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_physics_stats(u, v, grid, metrics=("gradsq",),
                          diffusivity=None, positive=False, eps=_EPS) -> dict:
    """
    Per-field physics metrics for one (field, timestep) pair. Same call shape
    as `compute_stats`. `metrics` selects from FIELD_METRICS.
    """
    u = u.double()
    v = v.double()
    if v.ndim == u.ndim:
        v = v.unsqueeze(0)
    c = FieldCtx(u, v, grid, diffusivity, positive, eps)
    out = {}
    for name in metrics:
        if name not in FIELD_METRICS:
            raise KeyError(f"unknown field metric '{name}'; "
                           f"available: {sorted(FIELD_METRICS)}")
        out |= FIELD_METRICS[name].fn(c)
    return out


def build_physics_field_df(truth, predictions, field_names, spec,
                           Ti=0, lead_times=None, traj_indices=None,
                           fields=None, params=None,
                           metrics=("gradsq",), eps=_EPS) -> pd.DataFrame:
    """
    One row per (model, traj, field, step). Same loop semantics as
    `build_stats_df`. `fields` restricts to a subset of channels;
    `metrics` restricts to a subset of FIELD_METRICS.
    """
    N_TRAJ, STEPS = truth.shape[:2]
    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))
    if fields is None:
        fields = list(field_names)

    pred_steps = [(p.shape[2] if p.ndim == 6 else p.shape[1])
                  for p in predictions.values()]
    max_steps = min(STEPS - Ti, *pred_steps)
    if lead_times is None:
        lead_times = list(range(max_steps))
    else:
        oob = [t for t in lead_times if t >= max_steps]
        if oob:
            raise ValueError(f"lead_times {oob} exceed available steps ({max_steps}).")

    n2i = {n: i for i, n in enumerate(field_names)}
    diff_map = spec.diffusivity or {}
    records = []

    for model_name, pred in predictions.items():
        is_ens = pred.ndim == 6
        for traj_idx in traj_indices:
            pars = _flat_params(params, traj_idx)
            for fname in fields:
                fi = n2i[fname]
                kkey = diff_map.get(fname)
                kappa = pars.get(kkey) if kkey else None
                positive = fname in spec.positive_fields
                for step in lead_times:
                    u = torch.as_tensor(truth[traj_idx, Ti + step, :, :, fi],
                                        dtype=torch.float32)
                    v = (pred[traj_idx, :, step, :, :, fi] if is_ens
                         else pred[traj_idx, step, :, :, fi].unsqueeze(0))
                    v = torch.as_tensor(v, dtype=torch.float32)
                    records.append({
                        "model": model_name, "traj": traj_idx,
                        "field": fname, "step": step,
                        **compute_physics_stats(u, v, spec.grid, metrics,
                                                kappa, positive, eps)})
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 -- global metrics
# ══════════════════════════════════════════════════════════════════════════════

def continuity_residual(f0, f1, dt, grid, idx) -> torch.Tensor:
    """
    RMS of d(rho)/dt + div(rho u), normalised by RMS of div(rho u).
    With no 'rho' role this reduces to ||div u|| / ||grad u||, so one equation
    covers all three datasets.

    The Well stores frames far more coarsely than the solver timestep (TRL:
    1.597 vs dt_sim ~ 1.5e-2), so the finite-difference time derivative is
    heavily discretisation-biased. Compute it identically on ground truth and
    report the RATIO -- the bias then cancels.
    """
    if "rho" not in idx:
        ux, uy = f1[..., idx["ux"]], f1[..., idx["uy"]]
        d = grid.d_dx(ux) + grid.d_dy(uy)
        g = (grid.d_dx(ux) ** 2 + grid.d_dy(ux) ** 2
             + grid.d_dx(uy) ** 2 + grid.d_dy(uy) ** 2)
        return (grid.avg(d ** 2) / (grid.avg(g) + _EPS)).sqrt()
    r0, r1 = f0[..., idx["rho"]], f1[..., idx["rho"]]
    ux = 0.5 * (f0[..., idx["ux"]] + f1[..., idx["ux"]])
    uy = 0.5 * (f0[..., idx["uy"]] + f1[..., idx["uy"]])
    rm = 0.5 * (r0 + r1)
    flux = grid.d_dx(rm * ux) + grid.d_dy(rm * uy)
    res = (r1 - r0) / dt + flux
    return (grid.avg(res ** 2) / (grid.avg(flux ** 2) + _EPS)).sqrt()


def compute_physics_global(u, v, spec, idx, params=None,
                           metrics=("kinetic_energy", "enstrophy"),
                           eps=_EPS) -> dict:
    """
    Global coupled diagnostics for one (state, timestep) pair.
    `metrics` selects from GLOBAL_METRICS; unknown names raise, and metrics
    whose required roles are absent are silently skipped.
    """
    c = Ctx(u.double(), v.double(), spec.grid, idx, dict(params or {}), eps)
    out = {}
    for name in metrics:
        if name == "continuity":
            continue                      # handled by the df builder
        if name not in GLOBAL_METRICS:
            raise KeyError(f"unknown global metric '{name}'; "
                           f"available: {sorted(GLOBAL_METRICS)}")
        m = GLOBAL_METRICS[name]
        if not set(m.roles) <= set(idx):
            continue
        out |= m.fn(c)
    return out


def build_physics_global_df(truth, predictions, field_names, spec,
                            Ti=0, lead_times=None, traj_indices=None,
                            params=None,
                            metrics=("kinetic_energy", "enstrophy"),
                            continuity=False, eps=_EPS) -> pd.DataFrame:
    """
    One row per (model, traj, step). Same loop semantics as `build_stats_df`,
    minus the `field` axis.

    `params` may be flat or keyed by trajectory index -- which you need for RB,
    where kappa and nu vary across the (Ra, Pr) settings.
    `continuity=True` adds the two-frame mass-conservation residual.
    """
    idx = spec.resolve(field_names)
    if "ux" not in idx or "uy" not in idx:
        raise KeyError(f"need 'ux' and 'uy' roles; resolved only {sorted(idx)}")

    metrics = list(metrics)
    if "continuity" in metrics:
        continuity = True
        metrics.remove("continuity")

    N_TRAJ, STEPS = truth.shape[:2]
    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))

    pred_steps = [(p.shape[2] if p.ndim == 6 else p.shape[1])
                  for p in predictions.values()]
    max_steps = min(STEPS - Ti, *pred_steps)
    if lead_times is None:
        lead_times = list(range(max_steps))
    else:
        oob = [t for t in lead_times if t >= max_steps]
        if oob:
            raise ValueError(f"lead_times {oob} exceed available steps ({max_steps}).")

    records = []
    for model_name, pred in predictions.items():
        is_ens = pred.ndim == 6
        for traj_idx in traj_indices:
            pars = _flat_params(params, traj_idx)
            for step in lead_times:
                u = torch.as_tensor(truth[traj_idx, Ti + step], dtype=torch.float32)
                v = (pred[traj_idx, :, step] if is_ens
                     else pred[traj_idx, step].unsqueeze(0))
                v = torch.as_tensor(v, dtype=torch.float32)

                stats = compute_physics_global(u, v, spec, idx, pars, metrics, eps)

                if continuity and step > 0:
                    u0 = torch.as_tensor(truth[traj_idx, Ti + step - 1],
                                         dtype=torch.float32).double()
                    v0 = (pred[traj_idx, :, step - 1] if is_ens
                          else pred[traj_idx, step - 1].unsqueeze(0))
                    v0 = torch.as_tensor(v0, dtype=torch.float32).double()
                    ct = float(continuity_residual(u0, u.double(), spec.dt,
                                                   spec.grid, idx))
                    cp = continuity_residual(v0, v.double(), spec.dt, spec.grid, idx)
                    stats["continuity_true"] = ct
                    stats["continuity_pred"] = float(cp.mean())
                    stats["continuity_ratio"] = float(cp.mean() / (ct + _EPS))

                records.append({"model": model_name, "traj": traj_idx,
                                "step": step, **stats})
    return pd.DataFrame(records)

"""Central problem-definition file â€” THE ONLY FILE YOU NEED TO EDIT
to switch governing equations, boundary conditions, or initial conditions.

Structure
---------
1. Declare the physics via helper dataclasses:
       BoundaryCondition  â€” one BC with a residual evaluator
       InitialCondition   â€” one IC with a residual evaluator (optional)
       DomainSpec         â€” coordinate / time bounds
       ProblemSpec        â€” assembles PDE + BCs + ICs into one object

2. Set   ACTIVE_PROBLEM = ProblemSpec(...)   at the bottom.

3. If your problem has no time dimension set  t_range=None  in DomainSpec
   and leave  initial_conditions=None  in ProblemSpec â€” both are optional.

Runtime parameters (e.g. a thermal-loading function computed before training)
can be injected without editing this file:
    from src.problem_definition import ACTIVE_PROBLEM
    ACTIVE_PROBLEM.set_param("tau0_fn", my_function)

Residual function signatures
-----------------------------
  PDE:   pde_fn(net, x1, x3, t)        -> list[Tensor]
  BC:    residual_fn(net, N, dev, dt, **params) -> Tensor | tuple[Tensor, ...]
  IC:    residual_fn(net, N, dev, dt, **params) -> Tensor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from math import gamma as _gamma_fn
from torch import Tensor


# â”€â”€ Dataclasses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class BoundaryCondition:
    """One boundary condition for the PINN.

    Parameters
    ----------
    name        : Human-readable label used in loss logs.
    residual_fn : Callable(net, N, device, dtype, **params) -> Tensor or
                  tuple[Tensor, ...].  Return a tuple when multiple
                  sub-conditions share one sampler (e.g. BC5 top+bottom).
    weight      : Per-BC loss weight multiplier (default 1.0).
    """
    name: str
    residual_fn: Callable
    weight: float = 1.0


@dataclass
class InitialCondition:
    """One initial condition for the PINN (optional â€” omit for steady problems).

    Parameters
    ----------
    name        : Human-readable label used in loss logs.
    residual_fn : Callable(net, N, device, dtype, **params) -> Tensor.
    weight      : Per-IC loss weight multiplier (default 1.0).
    """
    name: str
    residual_fn: Callable
    weight: float = 1.0


@dataclass
class DomainSpec:
    """Coordinate and time bounds for collocation point sampling.

    Set t_range=None for purely spatial (steady / quasi-static) problems.
    Set y_range for 3-D problems; leave None for 2-D (x, z) problems.
    """
    x1_range: tuple[float, float]
    x3_range: tuple[float, float]
    t_range:  Optional[tuple[float, float]] = None  # None â†’ time-independent
    y_range:  Optional[tuple[float, float]] = None  # None â†’ 2-D problem


@dataclass
class ProblemSpec:
    """Complete PINN problem specification.

    Attributes
    ----------
    name                : Human-readable problem name.
    domain              : DomainSpec â€” coordinate bounds.
    pde_fn              : Callable(net, x1, x3, t) -> list[Tensor] of residuals.
                          Each element is one PDE residual (N,1) tensor.
    boundary_conditions : Ordered list of BoundaryCondition objects.
    initial_conditions  : Optional list of InitialCondition objects.
                          Pass None or [] for time-independent problems.
    params              : Dict of mutable runtime parameters injected into
                          residual callables as keyword arguments.
                          Update via  set_param(key, value).
    """
    name:                str
    domain:              DomainSpec
    pde_fn:              Callable
    boundary_conditions: list[BoundaryCondition]
    initial_conditions:  Optional[list[InitialCondition]] = None
    params:              dict = field(default_factory=dict)

    def set_param(self, key: str, value) -> None:
        """Inject / update a runtime parameter without editing this file.

        Example::
            ACTIVE_PROBLEM.set_param("tau0_fn", thermal_loading_fn)
        """
        self.params[key] = value

    @property
    def has_initial_conditions(self) -> bool:
        """True when at least one InitialCondition is registered."""
        return bool(self.initial_conditions)

    @property
    def is_time_dependent(self) -> bool:
        """True when the domain includes a time axis."""
        return self.domain.t_range is not None


# =============================================================================
# 3-D PIEZOELECTRIC THERMOELASTIC PROBLEM (FRACTIONAL-ORDER HEAT CONDUCTION)
#
# Domain:   x âˆˆ [0, a],  y âˆˆ [0, b],  z âˆˆ [0, h]
# Unknowns: u(x,y,z),  v(x,y,z),  w(x,y,z),  Ï†(x,y,z),  T(x,y,z)
#
# Governing equations (5 PDEs):
#   Eq 1 â€” u-momentum
#   Eq 2 â€” v-momentum
#   Eq 3 â€” w-momentum
#   Eq 4 â€” Gauss law  (âˆ‡Â·D = 0)
#   Eq 5 â€” Fractional-order heat conduction (Riemannâ€“Liouville)
#
# HOW TO USE:
#   Fill in material constants in MATERIAL below.
# =============================================================================


# â”€â”€ 1. Material / problem constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â˜… Fill in your material constants in MATERIAL below â˜…

@dataclass
class PiezoThermConfig:
    """Material and geometry constants for the 3-D piezoelectric-thermoelastic
    problem with fractional-order heat conduction."""
    # Elastic stiffness (transversely isotropic)
    C11: float = 1.0;  C12: float = 0.3;  C13: float = 0.2
    C33: float = 1.1;  C44: float = 0.4
    # Piezoelectric coefficients
    e15: float = 0.1;  e31: float = 0.05;  e33: float = 0.2
    # Dielectric permittivity (Ï„â‚ƒâ‚ƒ, Ï„â‚â‚)
    tau33: float = 1.0;  eps11: float = 0.8
    # Thermal conductivity
    K11: float = 1.0;  K33: float = 1.0
    # Thermal stress coupling coefficients
    beta1: float = 0.1;  beta3: float = 0.1
    # Density, specific heat, reference temperature
    rho: float = 1.0;  ce: float = 1.0;  T0: float = 1.0
    # Pyroelectric coefficient
    P3: float = 0.05
    # Geometry
    a: float = 1.0   # x âˆˆ [0, a]
    b: float = 1.0   # y âˆˆ [0, b]
    h: float = 1.0   # z âˆˆ [0, h]
    # Wave parameter
    c: float = 1.0
    # Fractional order  (0 < alpha < 1)
    alpha: float = 0.5
    # Thermal relaxation time
    tau0: float = 0.1
    # Gaussâ€“Legendre quadrature points for Râ€“L fractional integral
    n_quad: int = 8
    # Boundary temperature values
    T1: float = 0.0;  T2: float = 0.0   # T at x=0, x=a
    T3: float = 0.0;  T4: float = 0.0   # T at y=0, y=b
    t1: float = 0.0;  t2: float = 0.0   # T at z=0, z=h
    # Boundary electric-potential values (normalised)
    phi1: float = 0.0;  phi2: float = 0.0   # Ï† at x=0, x=a
    phi3: float = 0.0;  phi4: float = 0.0   # Ï† at y=0, y=b
    g1:   float = 0.0;  g2:   float = 0.0   # Ï† at z=0, z=h


MATERIAL = PiezoThermConfig()   # â† edit values here


# â”€â”€ 2. Network â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PiezoThermNet(nn.Module):
    """5-output PINN: inputs (x, y, z) â†’ outputs (u, v, w, Ï†, T)."""

    def __init__(self, layers: list[int] | None = None):
        super().__init__()
        if layers is None:
            layers = [3, 64, 64, 64, 5]
        net_layers: list[nn.Module] = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net_layers.append(nn.Tanh())
        self.net = nn.Sequential(*net_layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> tuple[Tensor, ...]:
        inp = torch.cat([x, y, z], dim=-1)
        out = self.net(inp)
        return (out[..., 0:1], out[..., 1:2], out[..., 2:3],
                out[..., 3:4], out[..., 4:5])


# â”€â”€ 3. Autograd helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _g(f: Tensor, v: Tensor) -> Tensor:
    """âˆ‚f/âˆ‚v"""
    return torch.autograd.grad(
        f, v, grad_outputs=torch.ones_like(f),
        create_graph=True, retain_graph=True,
    )[0]


def _d2(f: Tensor, v1: Tensor, v2: Tensor) -> Tensor:
    """âˆ‚Â²f / âˆ‚v1 âˆ‚v2"""
    return _g(_g(f, v1), v2)


# â”€â”€ 4. Î¨ helper (heat equation inner expression) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _compute_psi(
    u: Tensor, v: Tensor, w: Tensor, phi: Tensor, T: Tensor,
    x: Tensor, y: Tensor, z: Tensor,
    m: "PiezoThermConfig",
) -> Tensor:
    """Î¨ = T + (Î²â‚h/Î²â‚ƒa) u_x + (Î²â‚h/Î²â‚ƒb) v_y + w_z âˆ’ (Pâ‚ƒeâ‚ƒâ‚ƒ/Ï„â‚ƒâ‚ƒÎ²â‚ƒ) Ï†_z  (Eq 5 RHS)"""
    return (
        T
        + (m.beta1 * m.h) / (m.beta3 * m.a) * _g(u, x)
        + (m.beta1 * m.h) / (m.beta3 * m.b) * _g(v, y)
        + _g(w, z)
        - (m.P3 * m.e33) / (m.tau33 * m.beta3) * _g(phi, z)
    )


# â”€â”€ 5. Fractional derivative  D_y^{1+Î±} Î¨  (Riemannâ€“Liouville / Caputo) â”€â”€â”€â”€â”€â”€

def _fractional_deriv(
    net: "PiezoThermNet",
    x: Tensor, y: Tensor, z: Tensor,
    m: "PiezoThermConfig",
) -> Tensor:
    """Caputo D_y^{1+Î±} Î¨ via Gaussâ€“Legendre quadrature on [0, y].

    D_y^{1+Î±} Î¨(y) = (y^{1-Î±}/Î“(1-Î±)) âˆ«â‚€Â¹ (1-s)^{-Î±} Î¨_ss(x, yÂ·s, z) ds
    """
    import numpy as np
    nq = m.n_quad
    xi, wi = np.polynomial.legendre.leggauss(nq)
    sk = torch.tensor(0.5 * (xi + 1.0), dtype=y.dtype, device=y.device)  # (nq,)
    wk = torch.tensor(0.5 * wi,         dtype=y.dtype, device=y.device)  # (nq,)

    gval  = _gamma_fn(1.0 - m.alpha)
    N     = y.shape[0]
    accum = torch.zeros(N, 1, dtype=y.dtype, device=y.device)

    for k in range(nq):
        p_k = (sk[k] * y).detach().requires_grad_(True)   # (N,1) quadrature abscissa
        x_k = x.detach().requires_grad_(True)
        z_k = z.detach().requires_grad_(True)
        u_, v_, w_, phi_, T_ = net(x_k, p_k, z_k)
        psi   = _compute_psi(u_, v_, w_, phi_, T_, x_k, p_k, z_k, m)
        psi_pp = _d2(psi, p_k, p_k)                       # âˆ‚Â²Î¨/âˆ‚pÂ²
        kernel = (1.0 - sk[k]).clamp(min=1e-12) ** (-m.alpha)
        accum  = accum + wk[k] * kernel * psi_pp

    prefactor = y.clamp(min=1e-12) ** (1.0 - m.alpha) / gval
    return prefactor * accum


# â”€â”€ 6. Five governing PDE residuals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _pde_fn(
    net: "PiezoThermNet",
    x: Tensor, y: Tensor, z: Tensor,
    t: Tensor,   # unused (steady problem) â€” kept for framework compatibility
) -> list[Tensor]:
    """Return [R1, R2, R3, R4, R5] each of shape (N,1).

    Equations from the paper:
      R1 â€” u-momentum  (Eq 1)
      R2 â€” v-momentum  (Eq 2)
      R3 â€” w-momentum  (Eq 3)
      R4 â€” Gauss law âˆ‡Â·D = 0  (Eq 4)
      R5 â€” Fractional heat conduction  (Eq 5)
    """
    m = MATERIAL
    u, v, w, phi, T = net(x, y, z)

    a = m.a;  b = m.b;  h = m.h
    h2a2 = (h / a) ** 2;  h2b2 = (h / b) ** 2;  h2ab = h ** 2 / (a * b)

    # â”€â”€ second-order spatial derivatives â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    u_xx = _d2(u, x, x);  u_yy = _d2(u, y, y);  u_zz = _d2(u, z, z)
    u_xy = _d2(u, x, y);  u_xz = _d2(u, x, z)

    v_xx = _d2(v, x, x);  v_yy = _d2(v, y, y);  v_zz = _d2(v, z, z)
    v_xy = _d2(v, x, y);  v_yz = _d2(v, y, z)

    w_xx = _d2(w, x, x);  w_yy = _d2(w, y, y);  w_zz = _d2(w, z, z)
    w_xz = _d2(w, x, z);  w_yz = _d2(w, y, z)

    phi_xx = _d2(phi, x, x);  phi_yy = _d2(phi, y, y);  phi_zz = _d2(phi, z, z)
    phi_xz = _d2(phi, x, z);  phi_yz = _d2(phi, y, z)

    T_x  = _g(T, x);   T_y  = _g(T, y);   T_z  = _g(T, z)
    T_xx = _d2(T, x, x);  T_yy = _d2(T, y, y);  T_zz = _d2(T, z, z)

    _Kc = m.K33 ** 2 * m.c / (m.rho * m.ce ** 2 * m.h ** 2)  # K33Â²c/(ÏceÂ²hÂ²)

    # â”€â”€ Eq 1 â€” u-momentum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    R1 = (
          m.C11 * h2a2 * u_xx
        + (0.25 * (m.C11 - m.C12) * h2b2 - _Kc) * u_yy
        + 0.5 * m.C44 * u_zz
        + (m.C12 + 0.25 * (m.C11 - m.C12)) * h2ab * v_xy
        + (m.C13 + 0.5 * m.C44) * (h / a) * w_xz
        + (m.e31 + m.e15) * (m.e33 / m.tau33) * (h / b) * phi_xz
        - (m.beta3 * m.beta1 * h) / (m.rho * m.ce * b) * m.T0 * T_x
    )

    # â”€â”€ Eq 2 â€” v-momentum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    R2 = (
          (0.25 * m.C11 + 0.75 * m.C12) * h2ab * u_xy
        + 0.25 * (m.C11 - m.C12) * h2a2 * v_xx
        + (m.C13 + 0.5 * m.C44) * (h ** 2 / a ** 2) * w_yz
        + (m.C11 * h2b2 - _Kc) * v_yy
        + 0.5 * m.C44 * v_zz
        + (m.e31 + m.e15) * (m.e33 / m.tau33) * (h / b) * phi_yz
        - (m.beta3 * m.beta1 * h) / (m.rho * m.ce * b) * m.T0 * T_y
    )

    # â”€â”€ Eq 3 â€” w-momentum â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    R3 = (
          (m.C13 + 0.5 * m.C44) * (h / a) * u_xz
        + (m.C13 + 0.5 * m.C44) * (h / a) * w_yz
        + 0.5 * m.C44 * h2a2 * w_xx
        + (0.5 * m.C44 * h2b2 - _Kc) * w_yy
        + m.C33 * w_zz
        + (m.e15 * m.e33 / m.tau33) * h2a2 * phi_xx
        + (m.e15 * m.e33 / m.tau33) * h2b2 * phi_yy
        + (m.e33 ** 2 / m.tau33) * phi_zz
        - (m.beta3 ** 2) / (m.rho * m.ce) * m.T0 * T_z
    )

    # â”€â”€ Eq 4 â€” Gauss law  âˆ‡Â·D = 0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    R4 = (
          (0.5 * m.e15 + m.e31) * (1.0 / a) * _d2(u, x, z)
        + (0.5 * m.e15 + m.e31) * (1.0 / b) * _d2(v, y, z)
        + 0.5 * m.e15 / a ** 2 * _d2(w, x, x)
        + 0.5 * m.e15 / b ** 2 * _d2(w, y, y)
        + m.e33 / h * _d2(w, z, z)
        - (m.eps11 * m.e33 / m.tau33) / a ** 2 * phi_xx
        - (m.eps11 * m.e33 / m.tau33) / b ** 2 * phi_yy
        - m.e33 / h ** 2 * phi_zz
        + (m.P3 * m.beta3) / (m.h ** 2 * m.rho * m.ce) * T_z
    )

    # â”€â”€ Eq 5 â€” Fractional-order heat conduction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    psi   = _compute_psi(u, v, w, phi, T, x, y, z, m)
    psi_y = _g(psi, y)
    frac  = _fractional_deriv(net, x, y, z, m)
    coeff = m.c ** (1.0 + m.alpha) * m.tau0 ** m.alpha / _gamma_fn(m.alpha + 1.0)

    R5 = (
          (m.K11 / m.K33) * (h2a2 * T_xx + h2b2 * T_yy)
        + T_zz
        - m.c * psi_y
        - coeff * frac
    )

    return [R1, R2, R3, R4, R5]


# â”€â”€ 7. Face samplers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _face(val_x, rnd_x, val_y, rnd_y, val_z, rnd_z, N, dev, dt):
    """Generic face sampler. Pass a scalar for fixed coords, True for random."""
    m = MATERIAL
    def _c(val, rnd, hi):
        if rnd:
            return torch.rand(N, 1, device=dev, dtype=dt) * hi
        return torch.full((N, 1), val, device=dev, dtype=dt)
    return _c(val_x, rnd_x, m.a), _c(val_y, rnd_y, m.b), _c(val_z, rnd_z, m.h)

def _face_x0(N, dev, dt): return _face(0.0, False, 0, True, 0, True, N, dev, dt)
def _face_xa(N, dev, dt): m=MATERIAL; return _face(m.a, False, 0, True, 0, True, N, dev, dt)
def _face_y0(N, dev, dt): return _face(0, True, 0.0, False, 0, True, N, dev, dt)
def _face_yb(N, dev, dt): m=MATERIAL; return _face(0, True, m.b, False, 0, True, N, dev, dt)
def _face_z0(N, dev, dt): return _face(0, True, 0, True, 0.0, False, N, dev, dt)
def _face_zh(N, dev, dt): m=MATERIAL; return _face(0, True, 0, True, m.h, False, N, dev, dt)


# â”€â”€ 8. Temperature Dirichlet BCs (Eqs 10â€“15) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _bc_T_x0(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_x0(N, dev, dt)
    _, _, _, _, T = net(x, y, z); return T - m.T1 / m.T0

def _bc_T_xa(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_xa(N, dev, dt)
    _, _, _, _, T = net(x, y, z); return T - m.T2 / m.T0

def _bc_T_y0(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_y0(N, dev, dt)
    _, _, _, _, T = net(x, y, z); return T - m.T3 / m.T0

def _bc_T_yb(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_yb(N, dev, dt)
    _, _, _, _, T = net(x, y, z); return T - m.T4 / m.T0

def _bc_T_z0(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_z0(N, dev, dt)
    _, _, _, _, T = net(x, y, z); return T - m.t1 / m.T0

def _bc_T_zh(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_zh(N, dev, dt)
    _, _, _, _, T = net(x, y, z); return T - m.t2 / m.T0


# â”€â”€ 9. Electric potential Dirichlet BCs (Eqs 16â€“21) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _bc_phi_x0(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_x0(N, dev, dt)
    _, _, _, phi, _ = net(x, y, z); return phi - m.phi1

def _bc_phi_xa(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_xa(N, dev, dt)
    _, _, _, phi, _ = net(x, y, z); return phi - m.phi2

def _bc_phi_y0(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_y0(N, dev, dt)
    _, _, _, phi, _ = net(x, y, z); return phi - m.phi3

def _bc_phi_yb(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_yb(N, dev, dt)
    _, _, _, phi, _ = net(x, y, z); return phi - m.phi4

def _bc_phi_z0(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_z0(N, dev, dt)
    _, _, _, phi, _ = net(x, y, z); return phi - m.g1

def _bc_phi_zh(net, N, dev, dt, **_):
    m = MATERIAL; x, y, z = _face_zh(N, dev, dt)
    _, _, _, phi, _ = net(x, y, z); return phi - m.g2


# â”€â”€ 10. Mechanical BCs at x = 0 and x = a â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _bc_sigma_x(net, N, dev, dt, **_):
    """Ïƒ_x = 0: h/a u_x + C12/C11 h/b v_y + C13/C11 w_z
                + e31e33/(Ï„33 C11) Ï†_z âˆ’ Î²1Î²3T0/(C11Ïce) T = 0  at x=0,a (Eq 22)"""
    m = MATERIAL
    res = []
    for fn in (_face_x0, _face_xa):
        x, y, z = fn(N, dev, dt)
        x = x.requires_grad_(True); y = y.requires_grad_(True); z = z.requires_grad_(True)
        u, v, w, phi, T = net(x, y, z)
        res.append(
            (h := m.h) / m.a * _g(u, x)
            + m.C12 / m.C11 * h / m.b * _g(v, y)
            + m.C13 / m.C11 * _g(w, z)
            + m.e31 * m.e33 / (m.tau33 * m.C11) * _g(phi, z)
            - m.beta1 * m.beta3 * m.T0 / (m.C11 * m.rho * m.ce) * T
        )
    return tuple(res)

def _bc_v_x(net, N, dev, dt, **_):
    """v = 0 at x = 0 and x = a (Eq 23)"""
    return tuple(net(*fn(N, dev, dt))[1] for fn in (_face_x0, _face_xa))

def _bc_w_x(net, N, dev, dt, **_):
    """w = 0 at x = 0 and x = a (Eq 24)"""
    return tuple(net(*fn(N, dev, dt))[2] for fn in (_face_x0, _face_xa))


# â”€â”€ 11. Mechanical BCs at y = 0 and y = b â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _bc_sigma_y(net, N, dev, dt, **_):
    """Ïƒ_y = 0: C12/C11 h/a u_x + h/b v_y + C13/C11 w_z
                + e31e33/(Ï„33 C11) Ï†_z âˆ’ Î²1Î²3T0/(C11Ïce) T = 0  at y=0,b (Eq 25)"""
    m = MATERIAL
    res = []
    for fn in (_face_y0, _face_yb):
        x, y, z = fn(N, dev, dt)
        x = x.requires_grad_(True); y = y.requires_grad_(True); z = z.requires_grad_(True)
        u, v, w, phi, T = net(x, y, z)
        res.append(
            m.C12 / m.C11 * m.h / m.a * _g(u, x)
            + m.h / m.b * _g(v, y)
            + m.C13 / m.C11 * _g(w, z)
            + m.e31 * m.e33 / (m.tau33 * m.C11) * _g(phi, z)
            - m.beta1 * m.beta3 * m.T0 / (m.C11 * m.rho * m.ce) * T
        )
    return tuple(res)

def _bc_u_y(net, N, dev, dt, **_):
    """u = 0 at y = 0 and y = b (Eq 26)"""
    return tuple(net(*fn(N, dev, dt))[0] for fn in (_face_y0, _face_yb))

def _bc_w_y(net, N, dev, dt, **_):
    """w = 0 at y = 0 and y = b (Eq 27)"""
    return tuple(net(*fn(N, dev, dt))[2] for fn in (_face_y0, _face_yb))


# â”€â”€ 12. Electric BC at x = 0 and x = a â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _bc_elec_x(net, N, dev, dt, **_):
    """h/a u_x + h/b v_y + e33/e31 w_z âˆ’ e33/e31 Ï†_z + P3T0Î²3/(Ïce e31) T = 0
    at x = 0 and x = a  (Eq 28)"""
    m = MATERIAL
    res = []
    for fn in (_face_x0, _face_xa):
        x, y, z = fn(N, dev, dt)
        x = x.requires_grad_(True); y = y.requires_grad_(True); z = z.requires_grad_(True)
        u, v, w, phi, T = net(x, y, z)
        res.append(
            m.h / m.a * _g(u, x)
            + m.h / m.b * _g(v, y)
            + m.e33 / m.e31 * _g(w, z)
            - m.e33 / m.e31 * _g(phi, z)
            + m.P3 * m.T0 * m.beta3 / (m.rho * m.ce * m.e31) * T
        )
    return tuple(res)


# â”€â”€ 13. Mechanical / electric BCs at z = 0 and z = h â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _bc_sigma_z(net, N, dev, dt, **_):
    """Ïƒ_z = 0: C13/C33 h/a u_x + C13/C33 h/b v_y + w_z
                + e33Â²/(Ï„33 C33) Ï†_z âˆ’ Î²1Î²3T0/(C33Ïce) T = 0  at z=0,h (Eq 29)"""
    m = MATERIAL
    res = []
    for fn in (_face_z0, _face_zh):
        x, y, z = fn(N, dev, dt)
        x = x.requires_grad_(True); y = y.requires_grad_(True); z = z.requires_grad_(True)
        u, v, w, phi, T = net(x, y, z)
        res.append(
            m.C13 / m.C33 * m.h / m.a * _g(u, x)
            + m.C13 / m.C33 * m.h / m.b * _g(v, y)
            + _g(w, z)
            + m.e33 ** 2 / (m.tau33 * m.C33) * _g(phi, z)
            - m.beta1 * m.beta3 * m.T0 / (m.C33 * m.rho * m.ce) * T
        )
    return tuple(res)

def _bc_sigma_xz(net, N, dev, dt, **_):
    """Ïƒ_xz = 0: u_z + h/a w_x + 2e15e33/(C44Ï„33) h/a Ï†_x = 0
    at z = 0 and z = h  (Eq 30)"""
    m = MATERIAL
    res = []
    for fn in (_face_z0, _face_zh):
        x, y, z = fn(N, dev, dt)
        x = x.requires_grad_(True); y = y.requires_grad_(True); z = z.requires_grad_(True)
        u, _, w, phi, _ = net(x, y, z)
        res.append(
            _g(u, z)
            + m.h / m.a * _g(w, x)
            + 2 * m.e15 * m.e33 / (m.C44 * m.tau33) * m.h / m.a * _g(phi, x)
        )
    return tuple(res)

def _bc_sigma_yz(net, N, dev, dt, **_):
    """Ïƒ_yz = 0: v_z + h/b w_y + 2e15e33/(C44Ï„33) h/b Ï†_y = 0
    at z = 0 and z = h  (Eq 31)"""
    m = MATERIAL
    res = []
    for fn in (_face_z0, _face_zh):
        x, y, z = fn(N, dev, dt)
        x = x.requires_grad_(True); y = y.requires_grad_(True); z = z.requires_grad_(True)
        _, v, w, phi, _ = net(x, y, z)
        res.append(
            _g(v, z)
            + m.h / m.b * _g(w, y)
            + 2 * m.e15 * m.e33 / (m.C44 * m.tau33) * m.h / m.b * _g(phi, y)
        )
    return tuple(res)


# â”€â”€ 14. Assemble problem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_m = MATERIAL

_PIEZO_THERM_PROBLEM = ProblemSpec(
    name="3-D Piezoelectric Thermoelastic â€” Fractional-Order Heat Conduction",

    domain=DomainSpec(
        x1_range=(0.0, _m.a),
        x3_range=(0.0, _m.h),
        t_range=None,
        y_range=(0.0, _m.b),   # triggers 3-D branch in loss.py
    ),

    pde_fn=_pde_fn,

    boundary_conditions=[
        # â”€â”€ Temperature Dirichlet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        BoundaryCondition("bc_T_x0",     _bc_T_x0,     weight=10.0),
        BoundaryCondition("bc_T_xa",     _bc_T_xa,     weight=10.0),
        BoundaryCondition("bc_T_y0",     _bc_T_y0,     weight=10.0),
        BoundaryCondition("bc_T_yb",     _bc_T_yb,     weight=10.0),
        BoundaryCondition("bc_T_z0",     _bc_T_z0,     weight=10.0),
        BoundaryCondition("bc_T_zh",     _bc_T_zh,     weight=10.0),
        # â”€â”€ Electric potential Dirichlet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        BoundaryCondition("bc_phi_x0",   _bc_phi_x0,   weight=10.0),
        BoundaryCondition("bc_phi_xa",   _bc_phi_xa,   weight=10.0),
        BoundaryCondition("bc_phi_y0",   _bc_phi_y0,   weight=10.0),
        BoundaryCondition("bc_phi_yb",   _bc_phi_yb,   weight=10.0),
        BoundaryCondition("bc_phi_z0",   _bc_phi_z0,   weight=10.0),
        BoundaryCondition("bc_phi_zh",   _bc_phi_zh,   weight=10.0),
        # â”€â”€ Mechanical BCs at x-faces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        BoundaryCondition("bc_sigma_x",  _bc_sigma_x,  weight=1.0),
        BoundaryCondition("bc_v_x",      _bc_v_x,      weight=1.0),
        BoundaryCondition("bc_w_x",      _bc_w_x,      weight=1.0),
        # â”€â”€ Mechanical BCs at y-faces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        BoundaryCondition("bc_sigma_y",  _bc_sigma_y,  weight=1.0),
        BoundaryCondition("bc_u_y",      _bc_u_y,      weight=1.0),
        BoundaryCondition("bc_w_y",      _bc_w_y,      weight=1.0),
        # â”€â”€ Electric BC at x-faces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        BoundaryCondition("bc_elec_x",   _bc_elec_x,   weight=1.0),
        # â”€â”€ Mechanical / electric BCs at z-faces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        BoundaryCondition("bc_sigma_z",  _bc_sigma_z,  weight=1.0),
        BoundaryCondition("bc_sigma_xz", _bc_sigma_xz, weight=1.0),
        BoundaryCondition("bc_sigma_yz", _bc_sigma_yz, weight=1.0),
    ],

    initial_conditions=None,
)


# â”€â”€ Active Problem â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ACTIVE_PROBLEM = _PIEZO_THERM_PROBLEM   # â† 3-D piezo-thermoelastic, fractional heat

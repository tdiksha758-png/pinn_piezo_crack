# Cleaned Python file
# Retained only the section corresponding to the equations and boundary conditions
# provided in the LaTeX file.

# =============================================================================
# TRICLINIC PIEZOELECTRIC INTERFACE PROBLEM — MODAL EXPANSION  (Problem 2)
# Two general triclinic piezoelectric half-spaces coupled at z = 0.
#
# Upper half-space (superscript 0): z > 0  —  4 partial-wave modes (nets 0–3)
# Lower half-space (superscript ′): z < 0  —  3 partial-wave modes (nets 4–6)
#
# Governing equations (28 PDEs total: 4 per mode × 7 modes):
#   Eq1(u₁), Eq2(u₂), Eq3(u₃), Eq4(φ)  — equations of motion + Gauss law
#
# Boundary conditions at interface z = 0  (sum-of-modes form, eqs. 23–28):
#   BC1–3  Σ u₁,u₂,u₃ upper = Σ u₁,u₂,u₃ lower     (eqs. 23–25)
#   BC4–5  Σ T₃₁,T₃₂  upper = Σ T₃₁,T₃₂  lower     (eqs. 26–27)
#   BC6    Σ φ         upper = Σ φ         lower     (eq.  28)
#
# HOW TO USE:
#   1. Fill in the material constants in UPPER_MATERIAL / LOWER_MATERIAL below.
#   2. Adjust domain bounds (_TRI_L, _TRI_H, _TRI_T_END) if needed.
#   3. Uncomment the ACTIVE_PROBLEM = _MULTIMODE_PROBLEM line at the bottom.
#      That is the ONLY line to change to switch problems.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor


# ── 0. Problem specification dataclasses ────────────────────────────────────

@dataclass
class DomainSpec:
    """Spatial-temporal domain specification."""
    x1_range: tuple[float, float]
    x3_range: tuple[float, float]
    t_range: tuple[float, float] | None = None


@dataclass
class BoundaryCondition:
    """Single boundary condition: name, residual function, and loss weight."""
    name: str
    residual_fn: Callable
    weight: float = 1.0


@dataclass
class InitialCondition:
    """Single initial condition: name, residual function, and loss weight."""
    name: str
    residual_fn: Callable
    weight: float = 1.0


@dataclass
class ProblemSpec:
    """Complete problem specification: PDEs, BCs, ICs, domain, parameters."""
    name: str
    domain: DomainSpec
    pde_fn: Callable
    boundary_conditions: list[BoundaryCondition] | None = None
    initial_conditions: list[InitialCondition] | None = None
    params: dict | None = None

    @property
    def has_initial_conditions(self) -> bool:
        """Check if problem has initial conditions."""
        return self.initial_conditions is not None and len(self.initial_conditions) > 0

    @property
    def is_time_dependent(self) -> bool:
        """Check if problem is time-dependent (has t_range)."""
        return self.domain.t_range is not None


# ── 1. Material constants ─────────────────────────────────────────────────────
# Edit UPPER_MATERIAL and LOWER_MATERIAL. All other code adapts automatically.

@dataclass
class MaterialConfig:
    """All elastic, pre-stress, piezoelectric, and dielectric constants for one
    triclinic piezoelectric half-space (Voigt notation).

    Stiffness indices: C_ij where Voigt maps 1↔11, 2↔22, 3↔33, 4↔23, 5↔13, 6↔12
    Pre-stress       : P11 (σ₁₁⁰), P33 (σ₃₃⁰)
    Piezoelectric    : e_kl  (stress-charge form, row = electric direction)
    Dielectric       : eps11, eps33  (permittivity components)
    Density          : rho  [kg/m³]
    """
    # ── Elastic stiffness ────────────────────────────────────────────────────
    C11: float = 0.0;  C13: float = 0.0;  C14: float = 0.0
    C15: float = 0.0;  C16: float = 0.0
    C31: float = 0.0;  C33: float = 0.0;  C34: float = 0.0
    C35: float = 0.0;  C36: float = 0.0
    C41: float = 0.0;  C43: float = 0.0;  C44: float = 0.0
    C45: float = 0.0;  C46: float = 0.0
    C53: float = 0.0;  C54: float = 0.0;  C55: float = 0.0;  C56: float = 0.0
    C63: float = 0.0;  C65: float = 0.0;  C66: float = 0.0
    # ── Pre-stress ───────────────────────────────────────────────────────────
    P11: float = 0.0;  P33: float = 0.0
    # ── Piezoelectric (e_kl, stress-charge, Voigt l-index) ───────────────────
    e11: float = 0.0;  e13: float = 0.0;  e14: float = 0.0
    e15: float = 0.0;  e16: float = 0.0
    e31: float = 0.0;  e33: float = 0.0;  e34: float = 0.0
    e35: float = 0.0;  e36: float = 0.0
    # ── Dielectric permittivity ──────────────────────────────────────────────
    eps11: float = 1.0;  eps33: float = 1.0
    # ── Density ──────────────────────────────────────────────────────────────
    rho: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  ★  FILL IN YOUR MATERIAL CONSTANTS HERE  ★
# ─────────────────────────────────────────────────────────────────────────────
UPPER_MATERIAL = MaterialConfig(
    # ── upper half-space (superscript 0) — replace 0.0 with actual values ──
    C11=1.0,  C13=0.0,  C14=0.0,  C15=0.0,  C16=0.0,
    C31=0.0,  C33=1.0,  C34=0.0,  C35=0.0,  C36=0.0,
    C41=0.0,  C43=0.0,  C44=1.0,  C45=0.0,  C46=0.0,
    C53=0.0,  C54=0.0,  C55=1.0,  C56=0.0,
    C63=0.0,  C65=0.0,  C66=1.0,
    P11=0.0,  P33=0.0,
    e11=0.0,  e13=0.0,  e14=0.0,  e15=0.1,  e16=0.0,
    e31=0.1,  e33=0.0,  e34=0.0,  e35=0.0,  e36=0.0,
    eps11=1.0,  eps33=1.0,
    rho=1.0,
)

LOWER_MATERIAL = MaterialConfig(
    # ── lower half-space (superscript ′) — replace 0.0 with actual values ──
    C11=1.2,  C13=0.0,  C14=0.0,  C15=0.0,  C16=0.0,
    C31=0.0,  C33=1.2,  C34=0.0,  C35=0.0,  C36=0.0,
    C41=0.0,  C43=0.0,  C44=1.2,  C45=0.0,  C46=0.0,
    C53=0.0,  C54=0.0,  C55=1.2,  C56=0.0,
    C63=0.0,  C65=0.0,  C66=1.2,
    P11=0.0,  P33=0.0,
    e11=0.0,  e13=0.0,  e14=0.0,  e15=0.15, e16=0.0,
    e31=0.15, e33=0.0,  e34=0.0,  e35=0.0,  e36=0.0,
    eps11=1.2,  eps33=1.2,
    rho=1.2,
)

# ── Domain bounds (edit here if needed) ──────────────────────────────────────
_TRI_L:     float = 10.0   # x  ∈ [0, L]
_TRI_H:     float =  5.0   # z  ∈ [0, H]  (depth from interface, both regions)
_TRI_T_END: float =  2.0   # t  ∈ [0, T_END]


# ── 2. Network architecture ───────────────────────────────────────────────────

class TriclinicNet(nn.Module):
    """4-output PINN network: inputs (x, z, t) → outputs (u₁, u₂, u₃, φ)."""

    def __init__(self, layers: list[int] | None = None):
        super().__init__()
        if layers is None:
            layers = [3, 64, 64, 64, 4]   # default: 3 hidden layers
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

    def forward(self, x: Tensor, z: Tensor, t: Tensor) -> tuple[Tensor, ...]:
        inp = torch.cat([x, z, t], dim=-1)
        out = self.net(inp)
        return out[..., 0:1], out[..., 1:2], out[..., 2:3], out[..., 3:4]


# ── 2b. Multi-mode network (7 partial waves: 4 upper + 3 lower) ───────────────

class MultiModeNet(nn.Module):
    """7 partial-wave PINN networks for the modal triclinic interface problem.

    Nets[0–3]: upper half-space modes  (material = UPPER_MATERIAL)
    Nets[4–6]: lower half-space modes  (material = LOWER_MATERIAL)

    Usage::
        net = MultiModeNet()
        trainer = Trainer(net)
    """

    UPPER_IDX: tuple[int, ...] = (0, 1, 2, 3)
    LOWER_IDX: tuple[int, ...] = (4, 5, 6)

    def __init__(self, layers: list[int] | None = None):
        super().__init__()
        self.nets = nn.ModuleList([TriclinicNet(layers) for _ in range(7)])


# ── 3. Autograd helper ────────────────────────────────────────────────────────

def _g(f: Tensor, v: Tensor) -> Tensor:
    """First-order partial derivative ∂f/∂v (scalar autograd)."""
    return torch.autograd.grad(
        f, v, grad_outputs=torch.ones_like(f),
        create_graph=True, retain_graph=True,
    )[0]


def _d2(f: Tensor, v1: Tensor, v2: Tensor) -> Tensor:
    """Mixed second-order partial ∂²f / ∂v1 ∂v2."""
    return _g(_g(f, v1), v2)


# ── 4. PDE residuals ──────────────────────────────────────────────────────────

def _triclinic_pde(
    u1: Tensor, u2: Tensor, u3: Tensor, phi: Tensor,
    x: Tensor, z: Tensor, t: Tensor,
    m: MaterialConfig,
) -> list[Tensor]:
    """Four governing equations for one triclinic piezoelectric half-space.

    Eq1 (u₁), Eq2 (u₂), Eq3 (u₃) — equations of motion
    Eq4 (φ)                        — Gauss law (quasi-static electric field)

    All coefficients read directly from MaterialConfig m.
    """
    # ── second-order spatial derivatives ─────────────────────────────────────
    u1_xx = _d2(u1, x, x);  u1_xz = _d2(u1, x, z);  u1_zz = _d2(u1, z, z)
    u2_xx = _d2(u2, x, x);  u2_xz = _d2(u2, x, z);  u2_zz = _d2(u2, z, z)
    u3_xx = _d2(u3, x, x);  u3_xz = _d2(u3, x, z);  u3_zz = _d2(u3, z, z)
    phi_xx = _d2(phi, x, x);  phi_xz = _d2(phi, x, z);  phi_zz = _d2(phi, z, z)

    # ── second-order time derivatives (dynamic problem) ───────────────────────
    u1_tt = _d2(u1, t, t)
    u2_tt = _d2(u2, t, t)
    u3_tt = _d2(u3, t, t)

    # ── Eq 1 (u₁ momentum) ───────────────────────────────────────────────────
    R1 = (
        (m.C11 + m.P11) * u1_xx  +  2*m.C15       * u1_xz  + (m.C55 + m.P33) * u1_zz
        + m.C16          * u2_xx  + (m.C14 + m.C56) * u2_xz  +  m.C54          * u2_zz
        + m.C15          * u3_xx  + (m.C13 + m.C55) * u3_xz  +  m.C53          * u3_zz
        + m.e11          * phi_xx + (m.e31 + m.e15) * phi_xz +  m.e35          * phi_zz
        - m.rho * u1_tt
    )

    # ── Eq 2 (u₂ momentum) ───────────────────────────────────────────────────
    R2 = (
          m.C16          * u1_xx  + (m.C56 + m.C41) * u1_xz  +  m.C45          * u1_zz
        + (m.C66 + m.P11) * u2_xx +  2*m.C46        * u2_xz  + (m.C44 + m.P33) * u2_zz
        + m.C65          * u3_xx  + (m.C63 + m.C45) * u3_xz  +  m.C43          * u3_zz
        + m.e16          * phi_xx + (m.e36 + m.e14) * phi_xz +  m.e34          * phi_zz
        - m.rho * u2_tt
    )

    # ── Eq 3 (u₃ momentum) ───────────────────────────────────────────────────
    R3 = (
          m.C15          * u1_xx  + (m.C55 + m.C31) * u1_xz  +  m.C35          * u1_zz
        + m.C56          * u2_xx  + (m.C54 + m.C36) * u2_xz  +  m.C34          * u2_zz
        + (m.C55 + m.P11) * u3_xx +  2*m.C53        * u3_xz  + (m.C33 + m.P33) * u3_zz
        + m.e15          * phi_xx + (m.e35 + m.e31) * phi_xz +  m.e33          * phi_zz
        - m.rho * u3_tt
    )

    # ── Eq 4 (Gauss law for electric field, quasi-static) ─────────────────────
    R4 = (
          m.e11          * u1_xx  + (m.e15 + m.e31) * u1_xz  +  m.e35          * u1_zz
        + m.e16          * u2_xx  + (m.e14 + m.e36) * u2_xz  +  m.e34          * u2_zz
        + m.e15          * u3_xx  + (m.e13 + m.e35) * u3_xz  +  m.e33          * u3_zz
        - m.eps11        * phi_xx
        - m.eps33        * phi_zz
    )

    return [R1, R2, R3, R4]


def _multimode_pde_fn(
    net: MultiModeNet,
    x: Tensor,
    z: Tensor,
    t: Tensor,
) -> list[Tensor]:
    """PDE residuals for all 7 partial-wave modes.

    Upper modes (nets 0–3): 4 PDEs × 4 modes = 16 residuals
    Lower modes (nets 4–6): 4 PDEs × 3 modes = 12 residuals
    Total: 28 residuals
    """
    t.requires_grad_(True)
    res: list[Tensor] = []
    for i in MultiModeNet.UPPER_IDX:
        u1, u2, u3, phi = net.nets[i](x, z, t)
        res += _triclinic_pde(u1, u2, u3, phi, x, z, t, UPPER_MATERIAL)
    for i in MultiModeNet.LOWER_IDX:
        u1, u2, u3, phi = net.nets[i](x, z, t)
        res += _triclinic_pde(u1, u2, u3, phi, x, z, t, LOWER_MATERIAL)
    return res


# ── 5. Interface boundary conditions at z = 0 ─────────────────────────────────
# Traction vector on z-normal surface (Voigt, 2D plane with u2 out-of-plane):
#   T₃₁ = C₅₁ u₁_x + C₅₅(u₁_z+u₃_x) + C₅₃ u₃_z + C₅₆ u₂_x + C₅₄ u₂_z + e₁₅ φ_x + e₃₅ φ_z
#   T₃₂ = C₄₁ u₁_x + C₄₅(u₁_z+u₃_x) + C₄₃ u₃_z + C₄₆ u₂_x + C₄₄ u₂_z + e₁₄ φ_x + e₃₄ φ_z

def _traction_at_interface(
    u1: Tensor, u2: Tensor, u3: Tensor, phi: Tensor,
    x: Tensor, z: Tensor,
    m: MaterialConfig,
) -> tuple[Tensor, Tensor]:
    """Compute T₃₁ and T₃₂ at any (x, z) using constitutive law."""
    u1_x = _g(u1, x);   u1_z = _g(u1, z)
    u2_x = _g(u2, x);   u2_z = _g(u2, z)
    u3_x = _g(u3, x);   u3_z = _g(u3, z)
    phi_x = _g(phi, x);  phi_z = _g(phi, z)

    T31 = (
        m.C15 * u1_x
        + m.C55 * (u1_z + u3_x)
        + m.C53 * u3_z
        + m.C56 * u2_x
        + m.C54 * u2_z
        + m.e15 * phi_x
        + m.e35 * phi_z
    )
    T32 = (
        m.C41 * u1_x
        + m.C45 * (u1_z + u3_x)
        + m.C43 * u3_z
        + m.C46 * u2_x
        + m.C44 * u2_z
        + m.e14 * phi_x
        + m.e34 * phi_z
    )
    return T31, T32


def _sample_interface(N: int, device, dtype):
    """Sample N points on the interface z = 0."""
    x = torch.rand(N, 1, device=device, dtype=dtype) * _TRI_L
    z = torch.zeros(N, 1, device=device, dtype=dtype)
    t = torch.rand(N, 1, device=device, dtype=dtype) * _TRI_T_END
    return x, z, t


# ── 5b. Sum-of-modes interface BCs (eqs. 23–28) ─────────────────────────────
# BC equations at z = 0:
#   u_j^(0)+u_j^(1)+u_j^(2)+u_j^(3) = u_j^(4)+u_j^(5)+u_j^(6)   j = 1,2,3
#   T31^(0)+...+T31^(3)              = T31^(4)+...+T31^(6)
#   T32^(0)+...+T32^(3)              = T32^(4)+...+T32^(6)
#   φ^(0)+...+φ^(3)                 = φ^(4)+...+φ^(6)

# ── BC1 — Displacement u₁ continuity (eq. 23) ────────────────────────────────
def _bc_u1_sum(net: MultiModeNet, N: int, device, dtype, **_) -> Tensor:
    """BC1 — u₁ sum at z = 0:
    u₁^(0)+u₁^(1)+u₁^(2)+u₁^(3) = u₁^(4)+u₁^(5)+u₁^(6)"""
    x, z, t = _sample_interface(N, device, dtype)
    outs = [net.nets[i](x, z, t) for i in range(7)]
    return (sum(outs[i][0] for i in MultiModeNet.UPPER_IDX)
            - sum(outs[i][0] for i in MultiModeNet.LOWER_IDX))


# ── BC2 — Displacement u₂ continuity (eq. 24) ────────────────────────────────
def _bc_u2_sum(net: MultiModeNet, N: int, device, dtype, **_) -> Tensor:
    """BC2 — u₂ sum at z = 0:
    u₂^(0)+u₂^(1)+u₂^(2)+u₂^(3) = u₂^(4)+u₂^(5)+u₂^(6)"""
    x, z, t = _sample_interface(N, device, dtype)
    outs = [net.nets[i](x, z, t) for i in range(7)]
    return (sum(outs[i][1] for i in MultiModeNet.UPPER_IDX)
            - sum(outs[i][1] for i in MultiModeNet.LOWER_IDX))


# ── BC3 — Displacement u₃ continuity (eq. 25) ────────────────────────────────
def _bc_u3_sum(net: MultiModeNet, N: int, device, dtype, **_) -> Tensor:
    """BC3 — u₃ sum at z = 0:
    u₃^(0)+u₃^(1)+u₃^(2)+u₃^(3) = u₃^(4)+u₃^(5)+u₃^(6)"""
    x, z, t = _sample_interface(N, device, dtype)
    outs = [net.nets[i](x, z, t) for i in range(7)]
    return (sum(outs[i][2] for i in MultiModeNet.UPPER_IDX)
            - sum(outs[i][2] for i in MultiModeNet.LOWER_IDX))


# ── BC4 — Traction T₃₁ continuity (eq. 26) ───────────────────────────────────
def _bc_T31_sum(net: MultiModeNet, N: int, device, dtype, **_) -> Tensor:
    """BC4 — T₃₁ sum at z = 0:
    T₃₁^(0)+T₃₁^(1)+T₃₁^(2)+T₃₁^(3) = T₃₁^(4)+T₃₁^(5)+T₃₁^(6)"""
    x, z, t = _sample_interface(N, device, dtype)
    x = x.requires_grad_(True);  z = z.requires_grad_(True)
    outs = [net.nets[i](x, z, t) for i in range(7)]

    def _t31(i):
        mat = UPPER_MATERIAL if i in MultiModeNet.UPPER_IDX else LOWER_MATERIAL
        return _traction_at_interface(*outs[i], x, z, mat)[0]

    return (sum(_t31(i) for i in MultiModeNet.UPPER_IDX)
            - sum(_t31(i) for i in MultiModeNet.LOWER_IDX))


# ── BC5 — Traction T₃₂ continuity (eq. 27) ───────────────────────────────────
def _bc_T32_sum(net: MultiModeNet, N: int, device, dtype, **_) -> Tensor:
    """BC5 — T₃₂ sum at z = 0:
    T₃₂^(0)+T₃₂^(1)+T₃₂^(2)+T₃₂^(3) = T₃₂^(4)+T₃₂^(5)+T₃₂^(6)"""
    x, z, t = _sample_interface(N, device, dtype)
    x = x.requires_grad_(True);  z = z.requires_grad_(True)
    outs = [net.nets[i](x, z, t) for i in range(7)]

    def _t32(i):
        mat = UPPER_MATERIAL if i in MultiModeNet.UPPER_IDX else LOWER_MATERIAL
        return _traction_at_interface(*outs[i], x, z, mat)[1]

    return (sum(_t32(i) for i in MultiModeNet.UPPER_IDX)
            - sum(_t32(i) for i in MultiModeNet.LOWER_IDX))


# ── BC6 — Electric potential continuity (eq. 28) ─────────────────────────────
def _bc_phi_sum(net: MultiModeNet, N: int, device, dtype, **_) -> Tensor:
    """BC6 — φ sum at z = 0:
    φ^(0)+φ^(1)+φ^(2)+φ^(3) = φ^(4)+φ^(5)+φ^(6)"""
    x, z, t = _sample_interface(N, device, dtype)
    outs = [net.nets[i](x, z, t) for i in range(7)]
    return (sum(outs[i][3] for i in MultiModeNet.UPPER_IDX)
            - sum(outs[i][3] for i in MultiModeNet.LOWER_IDX))
def forward(self, x, z, t):

    outputs = []

    for net in self.nets:

        outputs.append(
            net(x, z, t)
        )

    return outputs

# ── 6. Assemble problem 2: triclinic modal expansion (7 partial waves) ─────────

_MULTIMODE_PROBLEM = ProblemSpec(
    name="Triclinic Piezoelectric Interface — Modal Expansion (7 Partial Waves)",

    domain=DomainSpec(
        x1_range=(0.0, _TRI_L),
        x3_range=(0.0, _TRI_H),
        t_range=(0.0, _TRI_T_END),
    ),

    pde_fn=_multimode_pde_fn,          # 28 PDEs (4 per mode × 7 modes)

    boundary_conditions=[
        BoundaryCondition("bc_u1_sum",  _bc_u1_sum,  weight=10.0),   # eq. 23
        BoundaryCondition("bc_u2_sum",  _bc_u2_sum,  weight=10.0),   # eq. 24
        BoundaryCondition("bc_u3_sum",  _bc_u3_sum,  weight=10.0),   # eq. 25
        BoundaryCondition("bc_T31_sum", _bc_T31_sum, weight=10.0),   # eq. 26
        BoundaryCondition("bc_T32_sum", _bc_T32_sum, weight=10.0),   # eq. 27
        BoundaryCondition("bc_phi_sum", _bc_phi_sum, weight=10.0),   # eq. 28
    ],

    initial_conditions=None,
)


# ── 7. Export active problem ──────────────────────────────────────────────────
# Change this line to switch between different problem configurations.
ACTIVE_PROBLEM = _MULTIMODE_PROBLEM


# ═════════════════════════════════════════════════════════════════════════════
# GOVERNING EQUATIONS (LATEX DOCUMENTATION)
# ═════════════════════════════════════════════════════════════════════════════
#
# UPPER HALF-SPACE (superscript 0, z > 0):
#
# Equation 1 (u₁ momentum):
#   (C₁₁⁰+P₁₁⁰)∂²u₁/∂x² + 2C₁₅⁰∂²u₁/∂x∂z + (C₅₅⁰+P₃₃⁰)∂²u₁/∂z²
#   + C₁₆⁰∂²u₂/∂x² + (C₁₄⁰+C₅₆⁰)∂²u₂/∂x∂z + C₅₄⁰∂²u₂/∂z²
#   + C₁₅⁰∂²u₃/∂x² + (C₁₃⁰+C₅₅⁰)∂²u₃/∂x∂z + C₅₃⁰∂²u₃/∂z²
#   + e₁₁⁰∂²φ/∂x² + (e₃₁⁰+e₁₅⁰)∂²φ/∂x∂z + e₃₅⁰∂²φ/∂z²
#   = ρ ∂²u₁/∂t²
#
# Equation 2 (u₂ momentum):
#   C₁₆⁰∂²u₁/∂x² + (C₅₆⁰+C₄₁⁰)∂²u₁/∂x∂z + C₄₅⁰∂²u₁/∂z²
#   + (C₆₆⁰+P₁₁⁰)∂²u₂/∂x² + 2C₄₆⁰∂²u₂/∂x∂z + (C₄₄⁰+P₃₃⁰)∂²u₂/∂z²
#   + C₆₅⁰∂²u₃/∂x² + (C₆₃⁰+C₄₅⁰)∂²u₃/∂x∂z + C₄₃⁰∂²u₃/∂z²
#   + e₁₆⁰∂²φ/∂x² + (e₃₆⁰+e₁₄⁰)∂²φ/∂x∂z + e₃₄⁰∂²φ/∂z²
#   = ρ ∂²u₂/∂t²
#
# Equation 3 (u₃ momentum):
#   C₁₅⁰∂²u₁/∂x² + (C₅₅⁰+C₃₁⁰)∂²u₁/∂x∂z + C₃₅⁰∂²u₁/∂z²
#   + C₅₆⁰∂²u₂/∂x² + (C₅₄⁰+C₃₆⁰)∂²u₂/∂x∂z + C₃₄⁰∂²u₂/∂z²
#   + (C₅₅⁰+P₁₁⁰)∂²u₃/∂x² + 2C₅₃⁰∂²u₃/∂x∂z + (C₃₃⁰+P₃₃⁰)∂²u₃/∂z²
#   + e₁₅⁰∂²φ/∂x² + (e₃₅⁰+e₃₁⁰)∂²φ/∂x∂z + e₃₃⁰∂²φ/∂z²
#   = ρ ∂²u₃/∂t²
#
# Equation 4 (Gauss law, quasi-static electric field):
#   e₁₁⁰∂²u₁/∂x² + (e₁₅⁰+e₃₁⁰)∂²u₁/∂x∂z + e₃₅⁰∂²u₁/∂z²
#   + e₁₆⁰∂²u₂/∂x² + (e₁₄⁰+e₃₆⁰)∂²u₂/∂x∂z + e₃₄⁰∂²u₂/∂z²
#   + e₁₅⁰∂²u₃/∂x² + (e₁₃⁰+e₃₅⁰)∂²u₃/∂x∂z + e₃₃⁰∂²u₃/∂z²
#   - ε₁₁⁰∂²φ/∂x² - ε₃₃⁰∂²φ/∂z² = 0
#
#
# LOWER HALF-SPACE (superscript ′, z < 0):
#
# Equation 5 (u₁ momentum):
#   (C₁₁′+P₁₁′)∂²u₁/∂x² + 2C₁₅′∂²u₁/∂x∂z + (C₅₅′+P₃₃′)∂²u₁/∂z²
#   + C₁₆′∂²u₂/∂x² + (C₁₄′+C₅₆′)∂²u₂/∂x∂z + C₅₄′∂²u₂/∂z²
#   + C₁₅′∂²u₃/∂x² + (C₁₃′+C₅₅′)∂²u₃/∂x∂z + C₅₃′∂²u₃/∂z²
#   + e₁₁′∂²φ/∂x² + (e₃₁′+e₁₅′)∂²φ/∂x∂z + e₃₅′∂²φ/∂z²
#   = ρ′ ∂²u₁/∂t²
#
# Equation 6 (u₂ momentum):
#   C₁₆′∂²u₁/∂x² + (C₅₆′+C₄₁′)∂²u₁/∂x∂z + C₄₅′∂²u₁/∂z²
#   + (C₆₆′+P₁₁′)∂²u₂/∂x² + 2C₄₆′∂²u₂/∂x∂z + (C₄₄′+P₃₃′)∂²u₂/∂z²
#   + C₆₅′∂²u₃/∂x² + (C₆₃′+C₄₅′)∂²u₃/∂x∂z + C₄₃′∂²u₃/∂z²
#   + e₁₆′∂²φ/∂x² + (e₃₆′+e₁₄′)∂²φ/∂x∂z + e₃₄′∂²φ/∂z²
#   = ρ′ ∂²u₂/∂t²
#
# Equation 7 (u₃ momentum):
#   C₁₅′∂²u₁/∂x² + (C₅₅′+C₃₁′)∂²u₁/∂x∂z + C₃₅′∂²u₁/∂z²
#   + C₅₆′∂²u₂/∂x² + (C₅₄′+C₃₆′)∂²u₂/∂x∂z + C₃₄′∂²u₂/∂z²
#   + (C₅₅′+P₁₁′)∂²u₃/∂x² + 2C₅₃′∂²u₃/∂x∂z + (C₃₃′+P₃₃′)∂²u₃/∂z²
#   + e₁₅′∂²φ/∂x² + (e₃₅′+e₃₁′)∂²φ/∂x∂z + e₃₃′∂²φ/∂z²
#   = ρ′ ∂²u₃/∂t²
#
# Equation 8 (Gauss law, quasi-static electric field):
#   e₁₁′∂²u₁/∂x² + (e₁₅′+e₃₁′)∂²u₁/∂x∂z + e₃₅′∂²u₁/∂z²
#   + e₁₆′∂²u₂/∂x² + (e₁₄′+e₃₆′)∂²u₂/∂x∂z + e₃₄′∂²u₂/∂z²
#   + e₁₅′∂²u₃/∂x² + (e₁₃′+e₃₅′)∂²u₃/∂x∂z + e₃₃′∂²u₃/∂z²
#   - ε₁₁′∂²φ/∂x² - ε₃₃′∂²φ/∂z² = 0
#
# ═════════════════════════════════════════════════════════════════════════════
# BOUNDARY CONDITIONS AT INTERFACE (z = 0)
# ═════════════════════════════════════════════════════════════════════════════
#
# BC1 (Displacement continuity u₁):
#   u₁⁽⁰⁾ + u₁⁽¹⁾ + u₁⁽²⁾ + u₁⁽³⁾ = u₁⁽⁴⁾ + u₁⁽⁵⁾ + u₁⁽⁶⁾
#
# BC2 (Displacement continuity u₂):
#   u₂⁽⁰⁾ + u₂⁽¹⁾ + u₂⁽²⁾ + u₂⁽³⁾ = u₂⁽⁴⁾ + u₂⁽⁵⁾ + u₂⁽⁶⁾
#
# BC3 (Displacement continuity u₃):
#   u₃⁽⁰⁾ + u₃⁽¹⁾ + u₃⁽²⁾ + u₃⁽³⁾ = u₃⁽⁴⁾ + u₃⁽⁵⁾ + u₃⁽⁶⁾
#
# BC4 (Stress continuity T₃₁):
#   T₃₁⁽⁰⁾ + T₃₁⁽¹⁾ + T₃₁⁽²⁾ + T₃₁⁽³⁾ = T₃₁⁽⁴⁾ + T₃₁⁽⁵⁾ + T₃₁⁽⁶⁾
#
# BC5 (Stress continuity T₃₂):
#   T₃₂⁽⁰⁾ + T₃₂⁽¹⁾ + T₃₂⁽²⁾ + T₃₂⁽³⁾ = T₃₂⁽⁴⁾ + T₃₂⁽⁵⁾ + T₃₂⁽⁶⁾
#
# BC6 (Electric potential continuity φ):
#   φ⁽⁰⁾ + φ⁽¹⁾ + φ⁽²⁾ + φ⁽³⁾ = φ⁽⁴⁾ + φ⁽⁵⁾ + φ⁽⁶⁾
#
# ═════════════════════════════════════════════════════════════════════════════



"""Boundary-condition residuals for the piezoelectric crack problem.

Domain: x₁ ∈ [0, L],  x₃ ∈ [0, H],  t ∈ [0, T_MAX]
Crack: x₁ = 0,  a < x₃ < b

BC1  (crack face, normal stress prescribed):
    τ₁₁(0, x₃, t) = −τ₀ − τ₀(x₃, t)          a < x₃ < b

BC2  (non-crack face, zero normal displacement):
    u₁(0, x₃, t) = 0                            0 ≤ x₃ ≤ a  and  b ≤ x₃ ≤ H

BC3  (full left face, zero shear stress):
    τ₁₃(0, x₃, t) = 0                           0 ≤ x₃ ≤ H

BC4  (full left face, zero normal electric displacement):
    D₁(0, x₃, t) = 0                            0 ≤ x₃ ≤ H

BC5  (top and bottom surfaces, zero shear/normal stress and D₃):
    τ₁₃(x₁, 0, t) = τ₁₃(x₁, H, t) = 0
    τ₃₃(x₁, 0, t) = τ₃₃(x₁, H, t) = 0
    D₃ (x₁, 0, t) = D₃ (x₁, H, t) = 0

BC_FAR (far field, x₁ → L):
    u₁ → 0,  u₃ → 0,  φ → 0
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import config as cfg
from .pde_residuals import first_order_derivatives, constitutive


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: sample boundary points
# ─────────────────────────────────────────────────────────────────────────────

def _linspace_t(N: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Uniform random time samples in [0, T_MAX]."""
    return torch.rand(N, 1, device=device, dtype=dtype) * cfg.T_MAX


def sample_left_face(N: int, device: torch.device, dtype: torch.dtype):
    """x₁ = 0,  x₃ ~ U[0, H],  t ~ U[0, T_MAX].  Returns (x1, x3, t)."""
    x1 = torch.zeros(N, 1, device=device, dtype=dtype)
    x3 = torch.rand(N, 1, device=device, dtype=dtype) * cfg.H
    t  = _linspace_t(N, device, dtype)
    return x1, x3, t


def sample_crack_face(N: int, device: torch.device, dtype: torch.dtype):
    """x₁ = 0,  x₃ ~ U[a, b],  t ~ U[0, T_MAX]."""
    x1 = torch.zeros(N, 1, device=device, dtype=dtype)
    x3 = cfg.A_CRACK + torch.rand(N, 1, device=device, dtype=dtype) * (
        cfg.B_CRACK - cfg.A_CRACK
    )
    t  = _linspace_t(N, device, dtype)
    return x1, x3, t


def sample_non_crack_face(N: int, device: torch.device, dtype: torch.dtype):
    """x₁ = 0,  x₃ in [0,a]∪[b,H],  t ~ U[0, T_MAX].
    Sample N//2 from each segment."""
    n1 = N // 2;  n2 = N - n1
    x1 = torch.zeros(N, 1, device=device, dtype=dtype)
    x3_lo = torch.rand(n1, 1, device=device, dtype=dtype) * cfg.A_CRACK
    x3_hi = cfg.B_CRACK + torch.rand(n2, 1, device=device, dtype=dtype) * (
        cfg.H - cfg.B_CRACK
    )
    x3 = torch.cat([x3_lo, x3_hi], dim=0)
    t  = _linspace_t(N, device, dtype)
    return x1, x3, t


def sample_top_bottom(N: int, device: torch.device, dtype: torch.dtype):
    """x₁ ~ U[0,L],  x₃ ∈ {0, H},  t ~ U[0, T_MAX].  N//2 per surface."""
    n = N // 2
    x1 = torch.rand(N, 1, device=device, dtype=dtype) * cfg.L_TRUNC
    x3_bot = torch.zeros(n, 1, device=device, dtype=dtype)
    x3_top = torch.full((N - n, 1), cfg.H, device=device, dtype=dtype)
    x3 = torch.cat([x3_bot, x3_top], dim=0)
    t  = _linspace_t(N, device, dtype)
    return x1, x3, t


def sample_far_field(N: int, device: torch.device, dtype: torch.dtype):
    """x₁ = L,  x₃ ~ U[0, H],  t ~ U[0, T_MAX]."""
    x1 = torch.full((N, 1), cfg.L_TRUNC, device=device, dtype=dtype)
    x3 = torch.rand(N, 1, device=device, dtype=dtype) * cfg.H
    t  = _linspace_t(N, device, dtype)
    return x1, x3, t


# ─────────────────────────────────────────────────────────────────────────────
# Residual functions
# ─────────────────────────────────────────────────────────────────────────────

def _require_grad(*tensors: Tensor) -> None:
    for t in tensors:
        t.requires_grad_(True)


def bc1_crack_normal_stress(
    net,
    tau0_fn,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """BC1: τ₁₁(0,x₃,t) + τ₀ + τ₀(x₃,t) = 0   on crack face."""
    x1, x3, t = sample_crack_face(N, device, dtype)
    _require_grad(x1, x3, t)

    u1, u3, phi = net(x1, x3, t)
    d = first_order_derivatives(u1, u3, phi, x1, x3)
    c = constitutive(d)

    # Evaluate thermal loading τ₀(x₃, t) at sample points (numpy interp)
    x3_np = x3.detach().cpu().numpy().ravel()
    t_np  = t.detach().cpu().numpy().ravel()
    tau0_thermal = torch.tensor(
        tau0_fn(x3_np, t_np).reshape(-1, 1),
        dtype=dtype, device=device,
    )

    tau0_const = cfg.TAU_0_CONST
    stress_ref = cfg.KAPPA_11 * cfg.T0_BC

    res = (
        c["tau11"]
        + tau0_const
        + tau0_thermal
    ) / stress_ref
    return res

def bc2_non_crack_displacement(
    net,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """BC2: u₁(0,x₃,t) = 0   on non-crack face."""
    x1, x3, t = sample_non_crack_face(N, device, dtype)
    u1, _u3, _phi = net(x1, x3, t)
    return u1


def bc3_left_shear_stress(
    net,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """BC3: τ₁₃(0,x₃,t) = 0   on full left face."""
    x1, x3, t = sample_left_face(N, device, dtype)
    _require_grad(x1, x3, t)

    u1, u3, phi = net(x1, x3, t)
    d = first_order_derivatives(u1, u3, phi, x1, x3)
    c = constitutive(d)
    stress_ref = cfg.KAPPA_11 * cfg.T0_BC
    return c["tau13"] / stress_ref

def bc4_left_D1(
    net,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """BC4: D₁(0,x₃,t) = 0   on full left face."""
    x1, x3, t = sample_left_face(N, device, dtype)
    _require_grad(x1, x3, t)

    u1, u3, phi = net(x1, x3, t)
    d = first_order_derivatives(u1, u3, phi, x1, x3)
    c = constitutive(d)
    D_ref = cfg.PHI_REF / cfg.H
    return c["D1"] / D_ref
   


def bc5_top_bottom(
    net,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """BC5: τ₁₃=0, τ₃₃=0, D₃=0   on x₃ ∈ {0, H}."""
    x1, x3, t = sample_top_bottom(N, device, dtype)
    _require_grad(x1, x3, t)

    u1, u3, phi = net(x1, x3, t)
    d = first_order_derivatives(u1, u3, phi, x1, x3)
    c = constitutive(d)
    stress_ref = cfg.KAPPA_11 * cfg.T0_BC
    D_ref = cfg.PHI_REF / cfg.H

    return (
        c["tau13"] / stress_ref,
        c["tau33"] / stress_ref,
        c["D3"] / D_ref,
    )


# def bc_far_field(
#     net,
#     N: int,
#     device: torch.device,
#     dtype: torch.dtype,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Far-field condition: u₁, u₃, φ → 0  as  x₁ → L."""
#     x1, x3, t = sample_far_field(N, device, dtype)
#     u1, u3, phi = net(x1, x3, t)
#     return u1, u3, phi

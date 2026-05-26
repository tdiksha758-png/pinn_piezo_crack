"""PDE residuals for the three governing equations of a pre-stressed
piezoelectric half-plane (PZT-4).

Governing equations (quasi-static, no body forces):

  Eq 1 (u₁):
    (μ₁₁+σ₁₁⁰) u₁,₁₁  +  (μ₄₄+σ₃₃⁰) u₁,₃₃
    + (μ₁₃+μ₄₄) u₃,₁₃  + (e₃₁+e₁₅) φ,₁₃  = 0

  Eq 2 (u₃):
    (μ₄₄+σ₁₁⁰) u₃,₁₁  +  (μ₃₃+σ₃₃⁰) u₃,₃₃
    + (μ₁₃+μ₄₄) u₁,₁₃  + e₁₅ φ,₁₁  + e₃₃ φ,₃₃  = 0

  Eq 3 (φ):
    e₁₅ u₃,₁₁  + e₃₃ u₃,₃₃  + (e₁₅+e₃₁) u₁,₁₃
    − ε₁₁ φ,₁₁  − ε₃₃ φ,₃₃  = 0

Comma-subscript denotes partial differentiation.
All second derivatives are computed via PyTorch autograd.

Constitutive relations (for BC computation):

    τ₁₁ = μ₁₁ u₁,₁  +  μ₁₃ u₃,₃  +  e₃₁ φ,₃
    τ₃₃ = μ₁₃ u₁,₁  +  μ₃₃ u₃,₃  +  e₃₃ φ,₃
    τ₁₃ = μ₄₄ (u₁,₃ + u₃,₁)  +  e₁₅ φ,₁
    D₁  = e₁₅ (u₁,₃ + u₃,₁)  −  ε₁₁ φ,₁
    D₃  = e₃₁ u₁,₁  +  e₃₃ u₃,₃  −  ε₃₃ φ,₃
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import config as cfg

# ── Convenience: scalar autograd helper ──────────────────────────────────────

def _grad(output: Tensor, inp: Tensor, create_graph: bool = True) -> Tensor:
    """∂output/∂inp  (element-wise, same shape as output/inp)."""
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
# First-order derivatives of (u₁, u₃, φ)
# ─────────────────────────────────────────────────────────────────────────────

def first_order_derivatives(
    u1: Tensor, u3: Tensor, phi: Tensor,
    x1: Tensor, x3: Tensor,
) -> dict[str, Tensor]:
    """Return all first-order partial derivatives needed for constitutive laws.

    Parameters
    ----------
    u1, u3, phi : network outputs  (N, 1)
    x1, x3      : input coordinates with requires_grad=True  (N, 1)
    """
    u1_x1 = _grad(u1, x1)
    u1_x3 = _grad(u1, x3)
    u3_x1 = _grad(u3, x1)
    u3_x3 = _grad(u3, x3)
    phi_x1 = _grad(phi, x1)
    phi_x3 = _grad(phi, x3)
    return {
        "u1_x1": u1_x1, "u1_x3": u1_x3,
        "u3_x1": u3_x1, "u3_x3": u3_x3,
        "phi_x1": phi_x1, "phi_x3": phi_x3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Constitutive relations  (τ, D)
# ─────────────────────────────────────────────────────────────────────────────

def constitutive(d: dict[str, Tensor]) -> dict[str, Tensor]:
    """Evaluate stress and electric displacement from first-order derivatives.

    Parameters
    ----------
    d : output of :func:`first_order_derivatives`
    """
    mu11  = cfg.MU_11;  mu13  = cfg.MU_13;  mu33  = cfg.MU_33
    mu44  = cfg.MU_44
    e31   = cfg.E_31;   e33   = cfg.E_33;   e15   = cfg.E_15
    eps11 = cfg.EPS_11; eps33 = cfg.EPS_33

    tau11 = mu11 * d["u1_x1"] + mu13 * d["u3_x3"] + e31 * d["phi_x3"]
    tau33 = mu13 * d["u1_x1"] + mu33 * d["u3_x3"] + e33 * d["phi_x3"]
    tau13 = mu44 * (d["u1_x3"] + d["u3_x1"]) + e15 * d["phi_x1"]
    D1    = e15  * (d["u1_x3"] + d["u3_x1"]) - eps11 * d["phi_x1"]
    D3    = e31  * d["u1_x1"] + e33 * d["u3_x3"] - eps33 * d["phi_x3"]

    return {"tau11": tau11, "tau33": tau33, "tau13": tau13, "D1": D1, "D3": D3}


# ─────────────────────────────────────────────────────────────────────────────
# PDE residuals
# ─────────────────────────────────────────────────────────────────────────────

def pde_residuals(
    u1: Tensor, u3: Tensor, phi: Tensor,
    x1: Tensor, x3: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute residuals of all three governing equations.

    Parameters
    ----------
    u1, u3, phi : network predictions at interior collocation points (N, 1)
    x1, x3      : collocation coordinates  (N, 1), requires_grad=True

    Returns
    -------
    R1, R2, R3 : residuals of Eq 1, Eq 2, Eq 3  — each (N, 1)
    """
    mu11  = cfg.MU_11;  mu13  = cfg.MU_13;  mu33  = cfg.MU_33
    mu44  = cfg.MU_44
    e31   = cfg.E_31;   e33   = cfg.E_33;   e15   = cfg.E_15
    eps11 = cfg.EPS_11; eps33 = cfg.EPS_33
    s11   = cfg.SIGMA_11_0; s33 = cfg.SIGMA_33_0

    # ── first derivatives ────────────────────────────────────────────────────
    u1_x1  = _grad(u1,  x1)
    u1_x3  = _grad(u1,  x3)
    u3_x1  = _grad(u3,  x1)
    u3_x3  = _grad(u3,  x3)
    phi_x1 = _grad(phi, x1)
    phi_x3 = _grad(phi, x3)

    # ── second derivatives ────────────────────────────────────────────────────
    u1_x1x1   = _grad(u1_x1,  x1)
    u1_x3x3   = _grad(u1_x3,  x3)
    u1_x1x3   = _grad(u1_x1,  x3)   # = ∂²u₁/∂x₁∂x₃
    u3_x1x1   = _grad(u3_x1,  x1)
    u3_x3x3   = _grad(u3_x3,  x3)
    u3_x1x3   = _grad(u3_x1,  x3)   # = ∂²u₃/∂x₁∂x₃
    phi_x1x1  = _grad(phi_x1, x1)
    phi_x3x3  = _grad(phi_x3, x3)
    phi_x1x3  = _grad(phi_x1, x3)   # = ∂²φ/∂x₁∂x₃

    # ── Eq 1 residual ─────────────────────────────────────────────────────────
    R1 = (
        ((mu11 + s11) * u1_x1x1
        + (mu44 + s33) * u1_x3x3
        + (mu13 + mu44) * u3_x1x3
        + (e31 + e15) * phi_x1x3)/(mu11 + s11)
    )

    # ── Eq 2 residual ─────────────────────────────────────────────────────────
    R2 = (
       ((mu44 + s11) * u3_x1x1
        + (mu33 + s33) * u3_x3x3
        + (mu13 + mu44) * u1_x1x3
        + e15 * phi_x1x1
        + e33 * phi_x3x3)/(mu44 + s11)
    )

    # ── Eq 3 residual ─────────────────────────────────────────────────────────
    R3 = (
       ( e15  * u3_x1x1
        + e33  * u3_x3x3
        + (e15 + e31) * u1_x1x3
        - eps11 * phi_x1x1
        - eps33 * phi_x3x3)/e15
    )

    return R1, R2, R3

"""Weighted total PINN loss for the piezoelectric crack problem.

Loss = w_PDE  · (‖R₁‖² + ‖R₂‖² + ‖R₃‖²)
     + w_BC   · (‖BC1‖² + ‖BC2‖² + ‖BC3‖² + ‖BC4‖²
                 + ‖BC5_τ₁₃‖² + ‖BC5_τ₃₃‖² + ‖BC5_D₃‖²)
     + w_FAR  · (‖u₁^far‖² + ‖u₃^far‖² + ‖φ^far‖²)

All squared norms are mean-squared values.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import config as cfg
from .pde_residuals import pde_residuals
from .boundary_conditions import (
    bc1_crack_normal_stress,
    bc2_non_crack_displacement,
    bc3_left_shear_stress,
    bc4_left_D1,
    bc5_top_bottom,
    bc_far_field,
)


def _mse(r: Tensor) -> Tensor:
    return (r ** 2).mean()


def pinn_loss(
    net,
    tau0_fn,
    N_int: int,
    N_bc: int,
    device: torch.device,
    dtype: torch.dtype,
    w_pde: float = cfg.W_PDE,
    w_bc:  float = cfg.W_BC,
    w_far: float = cfg.W_FAR,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute total PINN loss and return component dictionary.

    Parameters
    ----------
    net      : MechanicsNet
    tau0_fn  : callable(x3_arr, t_arr) → τ₀(x₃, t)  (from thermal_loading)
    N_int    : number of interior collocation points
    N_bc     : number of points per boundary segment
    device, dtype : torch device / dtype
    w_pde, w_bc, w_far : loss weights

    Returns
    -------
    total_loss : scalar Tensor
    components : dict with individual loss terms (for logging)
    """
    # ── Interior PDE points ──────────────────────────────────────────────────
    x1_int = torch.rand(N_int, 1, device=device, dtype=dtype,
                        requires_grad=True) * cfg.L_TRUNC
    x3_int = torch.rand(N_int, 1, device=device, dtype=dtype,
                        requires_grad=True) * cfg.H
    t_int  = torch.rand(N_int, 1, device=device, dtype=dtype) * cfg.T_MAX

    u1_int, u3_int, phi_int = net(x1_int, x3_int, t_int)
    R1, R2, R3 = pde_residuals(u1_int, u3_int, phi_int, x1_int, x3_int)

    loss_pde = _mse(R1) + _mse(R2) + _mse(R3)

    # ── Boundary conditions ──────────────────────────────────────────────────
    # BC1
    r_bc1 = bc1_crack_normal_stress(net, tau0_fn, N_bc, device, dtype)
    loss_bc1 = _mse(r_bc1)

    # BC2
    r_bc2 = bc2_non_crack_displacement(net, N_bc, device, dtype)
    loss_bc2 = _mse(r_bc2)

    # BC3
    r_bc3 = bc3_left_shear_stress(net, N_bc, device, dtype)
    loss_bc3 = _mse(r_bc3)

    # BC4
    r_bc4 = bc4_left_D1(net, N_bc, device, dtype)
    loss_bc4 = _mse(r_bc4)

    # BC5
    r_bc5_t13, r_bc5_t33, r_bc5_D3 = bc5_top_bottom(net, N_bc, device, dtype)
    loss_bc5 = _mse(r_bc5_t13) + _mse(r_bc5_t33) + _mse(r_bc5_D3)

    loss_bc = loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4 + loss_bc5

    # ── Far-field condition ───────────────────────────────────────────────────
    u1_far, u3_far, phi_far = bc_far_field(net, N_bc, device, dtype)
    loss_far = _mse(u1_far) + _mse(u3_far) + _mse(phi_far)

    # ── Total ────────────────────────────────────────────────────────────────
    total = w_pde * loss_pde + w_bc * loss_bc + w_far * loss_far

    components = {
        "pde":  loss_pde.detach(),
        "bc1":  loss_bc1.detach(),
        "bc2":  loss_bc2.detach(),
        "bc3":  loss_bc3.detach(),
        "bc4":  loss_bc4.detach(),
        "bc5":  loss_bc5.detach(),
        "far":  loss_far.detach(),
        "total": total.detach(),
    }
    return total, components

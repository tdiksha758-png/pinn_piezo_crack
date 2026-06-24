from __future__ import annotations

import torch
from torch import Tensor

from . import config as cfg


# ============================================================
# AUTOGRAD HELPER
# ============================================================

def _grad(output: Tensor, inp: Tensor, create_graph=True):
    return torch.autograd.grad(
        output,
        inp,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


# ============================================================
# PDE RESIDUALS
# ============================================================

def pde_residuals(
    u1: Tensor,
    u2: Tensor,
    u3: Tensor,
    phi: Tensor,
    x1: Tensor,
    x3: Tensor,
    t: Tensor,
):
    """
    Returns:
        R1 : u1 equation
        R2 : u2 equation
        R3 : u3 equation
        R4 : electric potential equation
    """

    # --------------------------------------------------------
    # First derivatives
    # --------------------------------------------------------

    u1_x1 = _grad(u1, x1)
    u1_x3 = _grad(u1, x3)

    u2_x1 = _grad(u2, x1)
    u2_x3 = _grad(u2, x3)

    u3_x1 = _grad(u3, x1)
    u3_x3 = _grad(u3, x3)

    phi_x1 = _grad(phi, x1)
    phi_x3 = _grad(phi, x3)

    u1_t = _grad(u1, t)
    u2_t = _grad(u2, t)
    u3_t = _grad(u3, t)

    # --------------------------------------------------------
    # Second derivatives
    # --------------------------------------------------------

    u1_x1x1 = _grad(u1_x1, x1)
    u1_x1x3 = _grad(u1_x1, x3)
    u1_x3x3 = _grad(u1_x3, x3)

    u2_x1x1 = _grad(u2_x1, x1)
    u2_x1x3 = _grad(u2_x1, x3)
    u2_x3x3 = _grad(u2_x3, x3)

    u3_x1x1 = _grad(u3_x1, x1)
    u3_x1x3 = _grad(u3_x1, x3)
    u3_x3x3 = _grad(u3_x3, x3)

    phi_x1x1 = _grad(phi_x1, x1)
    phi_x1x3 = _grad(phi_x1, x3)
    phi_x3x3 = _grad(phi_x3, x3)

    u1_tt = _grad(u1_t, t)
    u2_tt = _grad(u2_t, t)
    u3_tt = _grad(u3_t, t)

    # --------------------------------------------------------
    # Material constants
    # --------------------------------------------------------

    C11 = cfg.C11
    C13 = cfg.C13
    C14 = cfg.C14
    C15 = cfg.C15
    C16 = cfg.C16

    C31 = cfg.C31
    C33 = cfg.C33
    C34 = cfg.C34
    C35 = cfg.C35
    C36 = cfg.C36

    C41 = cfg.C41
    C43 = cfg.C43
    C44 = cfg.C44
    C45 = cfg.C45
    C46 = cfg.C46

    C53 = cfg.C53
    C54 = cfg.C54
    C55 = cfg.C55
    C56 = cfg.C56

    C63 = cfg.C63
    C65 = cfg.C65
    C66 = cfg.C66

    P11 = cfg.P11
    P33 = cfg.P33

    e11 = cfg.E11
    e13 = cfg.E13
    e14 = cfg.E14
    e15 = cfg.E15
    e16 = cfg.E16

    e31 = cfg.E31
    e33 = cfg.E33
    e34 = cfg.E34
    e35 = cfg.E35
    e36 = cfg.E36

    eps11 = cfg.EPS11
    eps33 = cfg.EPS33

    rho = cfg.RHO

    # ========================================================
    # Eq (8)
    # ========================================================

    R1 = (
        (C11 + P11) * u1_x1x1
        + 2.0 * C15 * u1_x1x3
        + (C55 + P33) * u1_x3x3
        + C16 * u2_x1x1
        + (C14 + C56) * u2_x1x3
        + C54 * u2_x3x3
        + C15 * u3_x1x1
        + (C13 + C55) * u3_x1x3
        + C53 * u3_x3x3
        + e11 * phi_x1x1
        + (e31 + e15) * phi_x1x3
        + e35 * phi_x3x3
        - rho * u1_tt
    )

    # ========================================================
    # Eq (9)
    # ========================================================

    R2 = (
        C16 * u1_x1x1
        + (C56 + C41) * u1_x1x3
        + C45 * u1_x3x3
        + (C66 + P11) * u2_x1x1
        + 2.0 * C46 * u2_x1x3
        + (C44 + P33) * u2_x3x3
        + C65 * u3_x1x1
        + (C63 + C45) * u3_x1x3
        + C43 * u3_x3x3
        + e16 * phi_x1x1
        + (e36 + e14) * phi_x1x3
        + e34 * phi_x3x3
        - rho * u2_tt
    )

    # ========================================================
    # Eq (10)
    # ========================================================

    R3 = (
        C15 * u1_x1x1
        + (C55 + C31) * u1_x1x3
        + C35 * u1_x3x3
        + C56 * u2_x1x1
        + (C54 + C36) * u2_x1x3
        + C34 * u2_x3x3
        + (C55 + P11) * u3_x1x1
        + 2.0 * C53 * u3_x1x3
        + (C33 + P33) * u3_x3x3
        + e15 * phi_x1x1
        + (e35 + e31) * phi_x1x3
        + e33 * phi_x3x3
        - rho * u3_tt
    )

    # ========================================================
    # Eq (11)
    # ========================================================

    R4 = (
        e11 * u1_x1x1
        + (e15 + e31) * u1_x1x3
        + e35 * u1_x3x3
        + e16 * u2_x1x1
        + (e14 + e36) * u2_x1x3
        + e34 * u2_x3x3
        + e15 * u3_x1x1
        + (e13 + e35) * u3_x1x3
        + e33 * u3_x3x3
        - eps11 * phi_x1x1
        - eps33 * phi_x3x3
    )

    return R1, R2, R3, R4
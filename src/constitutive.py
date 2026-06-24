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
# FIRST ORDER DERIVATIVES
# ============================================================

def first_order_derivatives(
    u1: Tensor,
    u2: Tensor,
    u3: Tensor,
    phi: Tensor,
    x1: Tensor,
    x3: Tensor,
):
    """
    Returns all first-order derivatives needed
    for constitutive relations.
    """

    u1_x1 = _grad(u1, x1)
    u1_x3 = _grad(u1, x3)

    u2_x1 = _grad(u2, x1)
    u2_x3 = _grad(u2, x3)

    u3_x1 = _grad(u3, x1)
    u3_x3 = _grad(u3, x3)

    phi_x1 = _grad(phi, x1)
    phi_x3 = _grad(phi, x3)

    return {
        "u1_x1": u1_x1,
        "u1_x3": u1_x3,
        "u2_x1": u2_x1,
        "u2_x3": u2_x3,
        "u3_x1": u3_x1,
        "u3_x3": u3_x3,
        "phi_x1": phi_x1,
        "phi_x3": phi_x3,
    }


# ============================================================
# CONSTITUTIVE RELATIONS
# ============================================================

def constitutive(d):
    """
    Stress components used in interface conditions.
    """

    # --------------------------------------------------------
    # 2D strain components
    # --------------------------------------------------------

    e11 = d["u1_x1"]

    e33 = d["u3_x3"]

    e13 = 0.5 * (
        d["u1_x3"]
        + d["u3_x1"]
    )

    # --------------------------------------------------------
    # Triclinic stress components
    # --------------------------------------------------------

    T31 = (
        cfg.C15 * e11
        + cfg.C35 * e33
        + cfg.C55 * e13
    )

    T32 = (
        cfg.C14 * e11
        + cfg.C34 * e33
        + cfg.C45 * e13
    )

    T33 = (
        cfg.C13 * e11
        + cfg.C33 * e33
        + cfg.C35 * e13
    )

    return {
        "T31": T31,
        "T32": T32,
        "T33": T33,
    }
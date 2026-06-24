from __future__ import annotations

import torch

from . import config as cfg

from .pde_residuals import pde_residuals

from .boundary_conditions import (
    bc_u1_continuity,
    bc_u2_continuity,
    bc_u3_continuity,
    bc_T31_continuity,
    bc_T32_continuity,
    bc_T33_continuity,
    bc_phi_continuity,
)


# ==========================================================
# MSE
# ==========================================================

def mse(x):
    return torch.mean(x**2)


# ==========================================================
# INTERIOR POINTS : LOWER LAYER
# ==========================================================

def sample_interior_lower(
    N,
    device,
    dtype,
):

    x1 = (
        cfg.X1_MIN
        + (cfg.X1_MAX - cfg.X1_MIN)
        * torch.rand(
            N, 1,
            device=device,
            dtype=dtype,
        )
    )

    x3 = (
        cfg.X3_MIN
        + (cfg.INTERFACE_Z - cfg.X3_MIN)
        * torch.rand(
            N, 1,
            device=device,
            dtype=dtype,
        )
    )

    t = (
        cfg.T_MIN
        + (cfg.T_MAX - cfg.T_MIN)
        * torch.rand(
            N, 1,
            device=device,
            dtype=dtype,
        )
    )

    x1.requires_grad_(True)
    x3.requires_grad_(True)
    t.requires_grad_(True)

    return x1, x3, t


# ==========================================================
# INTERIOR POINTS : UPPER LAYER
# ==========================================================

def sample_interior_upper(
    N,
    device,
    dtype,
):

    x1 = (
        cfg.X1_MIN
        + (cfg.X1_MAX - cfg.X1_MIN)
        * torch.rand(
            N, 1,
            device=device,
            dtype=dtype,
        )
    )

    x3 = (
        cfg.INTERFACE_Z
        + (cfg.X3_MAX - cfg.INTERFACE_Z)
        * torch.rand(
            N, 1,
            device=device,
            dtype=dtype,
        )
    )

    t = (
        cfg.T_MIN
        + (cfg.T_MAX - cfg.T_MIN)
        * torch.rand(
            N, 1,
            device=device,
            dtype=dtype,
        )
    )

    x1.requires_grad_(True)
    x3.requires_grad_(True)
    t.requires_grad_(True)

    return x1, x3, t


# ==========================================================
# TOTAL LOSS
# ==========================================================

def pinn_loss(
    net_lower,
    net_upper,
    N_int=1000,
    N_interface=300,
    device=None,
    dtype=torch.float64,
):

    if device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # ======================================================
    # LOWER LAYER
    # ======================================================

    x1_l, x3_l, t_l = sample_interior_lower(
        N_int,
        device,
        dtype,
    )

    u1_l, u2_l, u3_l, phi_l = net_lower(
        x1_l,
        x3_l,
        t_l,
    )

    R1_l, R2_l, R3_l, R4_l = pde_residuals(
        u1_l,
        u2_l,
        u3_l,
        phi_l,
        x1_l,
        x3_l,
        t_l,
    )

    loss_pde_lower = (
        mse(R1_l)
        + mse(R2_l)
        + mse(R3_l)
        + mse(R4_l)
    )

    # ======================================================
    # UPPER LAYER
    # ======================================================

    x1_u, x3_u, t_u = sample_interior_upper(
        N_int,
        device,
        dtype,
    )

    u1_u, u2_u, u3_u, phi_u = net_upper(
        x1_u,
        x3_u,
        t_u,
    )

    R1_u, R2_u, R3_u, R4_u = pde_residuals(
        u1_u,
        u2_u,
        u3_u,
        phi_u,
        x1_u,
        x3_u,
        t_u,
    )

    loss_pde_upper = (
        mse(R1_u)
        + mse(R2_u)
        + mse(R3_u)
        + mse(R4_u)
    )

    # ======================================================
    # INTERFACE CONDITIONS
    # ======================================================

    loss_interface = (

        mse(
            bc_u1_continuity(
                net_lower,
                net_upper,
                N_interface,
                device,
                dtype,
            )
        )

        + mse(
            bc_u2_continuity(
                net_lower,
                net_upper,
                N_interface,
                device,
                dtype,
            )
        )

        + mse(
            bc_u3_continuity(
                net_lower,
                net_upper,
                N_interface,
                device,
                dtype,
            )
        )

        + mse(
            bc_T31_continuity(
                net_lower,
                net_upper,
                N_interface,
                device,
                dtype,
            )
        )

        + mse(
            bc_T32_continuity(
                net_lower,
                net_upper,
                N_interface,
                device,
                dtype,
            )
        )

        + mse(
            bc_T33_continuity(
                net_lower,
                net_upper,
                N_interface,
                device,
                dtype,
            )
        )

        + mse(
            bc_phi_continuity(
                net_lower,
                net_upper,
                N_interface,
                device,
                dtype,
            )
        )
    )

    # ======================================================
    # TOTAL LOSS
    # ======================================================

    total_loss = (
        loss_pde_lower
        + loss_pde_upper
        + loss_interface
    )

    return total_loss

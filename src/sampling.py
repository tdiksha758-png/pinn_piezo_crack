"""
Sampling utilities for the thermo-piezoelectric crack PINN.

Domain:
    x1 ∈ [0, L_TRUNC]
    x3 ∈ [0, H]
    t  ∈ [0, T_MAX]
"""

from __future__ import annotations

import torch

from . import config as cfg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def sample_uniform(n, low, high, device=DEVICE, dtype=torch.float64):

    return low + (high - low) * torch.rand(
        n, 1,
        device=device,
        dtype=dtype
    )


# ─────────────────────────────────────────────────────────────
# Interior domain sampling
# ─────────────────────────────────────────────────────────────

def sample_domain_points(
    n_points,
    device=DEVICE,
    dtype=torch.float64,
):
    """
    Interior collocation points.

    Returns:
        x1, x3, t
    """

    x1 = sample_uniform(n_points, 0.0, cfg.L_TRUNC, device, dtype)
    x3 = sample_uniform(n_points, 0.0, cfg.H, device, dtype)
    t  = sample_uniform(n_points, 0.0, cfg.T_MAX, device, dtype)

    x1.requires_grad_(True)
    x3.requires_grad_(True)
    t.requires_grad_(True)

    return x1, x3, t


# ─────────────────────────────────────────────────────────────
# Left boundary x1 = 0
# ─────────────────────────────────────────────────────────────

def sample_left_face(
    n_points,
    device=DEVICE,
    dtype=torch.float64,
):

    x1 = torch.zeros(
        n_points, 1,
        device=device,
        dtype=dtype,
        requires_grad=True
    )

    x3 = sample_uniform(n_points, 0.0, cfg.H, device, dtype)

    t = sample_uniform(n_points, 0.0, cfg.T_MAX, device, dtype)

    x3.requires_grad_(True)
    t.requires_grad_(True)

    return x1, x3, t


# ─────────────────────────────────────────────────────────────
# Crack face
# ─────────────────────────────────────────────────────────────

def sample_crack_face(
    n_points,
    device=DEVICE,
    dtype=torch.float64,
):

    x1 = torch.zeros(
        n_points, 1,
        device=device,
        dtype=dtype,
        requires_grad=True
    )

    x3 = sample_uniform(
        n_points,
        cfg.A_CRACK,
        cfg.B_CRACK,
        device,
        dtype
    )

    t = sample_uniform(
        n_points,
        0.0,
        cfg.T_MAX,
        device,
        dtype
    )

    x3.requires_grad_(True)
    t.requires_grad_(True)

    return x1, x3, t


# ─────────────────────────────────────────────────────────────
# Top-bottom boundaries
# ─────────────────────────────────────────────────────────────

def sample_top_bottom(
    n_points,
    device=DEVICE,
    dtype=torch.float64,
):

    n_half = n_points // 2

    x1 = sample_uniform(
        n_points,
        0.0,
        cfg.L_TRUNC,
        device,
        dtype
    )

    x3_bottom = torch.zeros(
        n_half, 1,
        device=device,
        dtype=dtype
    )

    x3_top = torch.full(
        (n_points - n_half, 1),
        cfg.H,
        device=device,
        dtype=dtype
    )

    x3 = torch.cat([x3_bottom, x3_top], dim=0)

    t = sample_uniform(
        n_points,
        0.0,
        cfg.T_MAX,
        device,
        dtype
    )

    x1.requires_grad_(True)
    x3.requires_grad_(True)
    t.requires_grad_(True)

    return x1, x3, t


# ─────────────────────────────────────────────────────────────
# Far-field boundary x1 = L
# ─────────────────────────────────────────────────────────────

def sample_far_field(
    n_points,
    device=DEVICE,
    dtype=torch.float64,
):

    x1 = torch.full(
        (n_points, 1),
        cfg.L_TRUNC,
        device=device,
        dtype=dtype,
    )

    x3 = sample_uniform(
        n_points,
        0.0,
        cfg.H,
        device,
        dtype
    )

    t = sample_uniform(
        n_points,
        0.0,
        cfg.T_MAX,
        device,
        dtype
    )

    x1.requires_grad_(True)
    x3.requires_grad_(True)
    t.requires_grad_(True)

    return x1, x3, t
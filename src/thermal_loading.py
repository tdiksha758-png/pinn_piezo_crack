"""Compute the thermal loading τ₀(x₃, t) = μ_E0 (A(t)x₃ + B(t)) − k_E0 T^(1)(x₃, t).

A(t) and B(t) are determined at every time instant by two integral constraints
(vanishing net force and moment from the thermal stress resultant):

    ∫₀ᴴ  τ₀(x₃, t) dx₃          = 0   →  force-free condition
    ∫₀ᴴ  x₃ τ₀(x₃, t) dx₃       = 0   →  moment-free condition

Substituting τ₀ = μ_E0(A x₃ + B) − k_E0 T:

    μ_E0 (A H²/2  + B H)         = k_E0 ∫₀ᴴ T(x₃,t) dx₃           (I)
    μ_E0 (A H³/3  + B H²/2)      = k_E0 ∫₀ᴴ x₃ T(x₃,t) dx₃        (II)

This is a 2×2 linear system:
    [ H²/2   H  ] [A]   k_E0     [  ∫T dx₃   ]
    [ H³/3  H²/2] [B] = ─────  · [ ∫x₃T dx₃  ]
                        μ_E0

Solved by direct inversion (closed-form).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from . import config as cfg


def compute_AB(
    x3_grid: np.ndarray,
    T_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute A(t) and B(t) for every time snapshot.

    Parameters
    ----------
    x3_grid : (N_x,) spatial grid (uniform, from :func:`solve_temperature`)
    T_field : (N_x, N_t) temperature field T^(1)(x₃, t)

    Returns
    -------
    A : (N_t,)
    B : (N_t,)
    """
    H = cfg.H
    mu_e0 = cfg.MU_E0
    k_e0  = cfg.K_E0

    # 2×2 coefficient matrix — built with the SAME trapezoidal quadrature
    # as I1/I2 so that M @ [A,B] = (k_e0/mu_e0)*[I1,I2] is consistent.
    #   Force row:   ∫(Ax₃+B)dx₃  → coefficients [∫x₃dx₃,  ∫dx₃]
    #   Moment row:  ∫x₃(Ax₃+B)dx₃ → coefficients [∫x₃²dx₃, ∫x₃dx₃]
    # Note: trapezoid of a linear function is exact, but ∫x₃² is O(dx²)
    # accurate.  Using trapezoidal here keeps M and I1/I2 on the same grid.
    J0 = trapezoid(np.ones_like(x3_grid), x3_grid)   # = H  (exact)
    J1 = trapezoid(x3_grid,               x3_grid)   # = H²/2 (exact)
    J2 = trapezoid(x3_grid ** 2,          x3_grid)   # ≈ H³/3 (numerical)
    M = np.array([
        [J1, J0],
        [J2, J1],
    ])
    M_inv = np.linalg.inv(M)          # 2×2 — cheap

    N_t = T_field.shape[1]
    A = np.zeros(N_t)
    B = np.zeros(N_t)

    for j in range(N_t):
        T_j = T_field[:, j]
        I1 = trapezoid(T_j, x3_grid)                  # ∫₀ᴴ T dx₃
        I2 = trapezoid(x3_grid * T_j, x3_grid)        # ∫₀ᴴ x₃ T dx₃
        rhs = (k_e0 / mu_e0) * np.array([I1, I2])
        AB  = M_inv @ rhs
        A[j] = AB[0]
        B[j] = AB[1]

    return A, B


def compute_tau0_field(
    x3_grid: np.ndarray,
    t_grid: np.ndarray,
    T_field: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Evaluate τ₀(x₃, t) on the full (x₃, t) grid.

    Returns
    -------
    tau0 : (N_x, N_t)  thermal loading term
    """
    mu_e0 = cfg.MU_E0
    k_e0  = cfg.K_E0

    # Shape: (N_x, N_t)  — broadcast x₃ column against time rows
    x3_col = x3_grid[:, np.newaxis]          # (N_x, 1)
    A_row  = A[np.newaxis, :]                # (1, N_t)
    B_row  = B[np.newaxis, :]

    tau0 = mu_e0 * (A_row * x3_col + B_row) - k_e0 * T_field
    return tau0


def make_tau0_interpolator(
    x3_grid: np.ndarray,
    t_grid: np.ndarray,
    tau0_field: np.ndarray,
):
    """Return a callable  τ₀(x₃_arr, t_arr)  via bilinear interpolation.

    Parameters
    ----------
    x3_grid, t_grid : 1-D arrays from the temperature solver
    tau0_field : (N_x, N_t)

    Returns
    -------
    callable(x3: ndarray, t: ndarray) -> ndarray
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (x3_grid, t_grid), tau0_field,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    def tau0_fn(x3: np.ndarray, t: np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.asarray(x3).ravel(), np.asarray(t).ravel()])
        return interp(pts)

    return tau0_fn

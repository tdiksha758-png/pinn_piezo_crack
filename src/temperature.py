"""Numerical solver for the 1-D fractional Cattaneo–Vernotte heat equation.

PDE (x₃ ∈ [0, H],  t ∈ [0, T_max]):

    ∂²T / ∂x₃²  =  (1/λ₀) [ ∂T/∂t  +  C_γ · ∂^(1+γ)T / ∂t^(1+γ) ]

where  C_γ = τ_q^γ / Γ(1+γ)  and  ∂^(1+γ)/∂t^(1+γ)  is the Caputo
fractional derivative of order β = 1+γ ∈ (1, 2].

When γ = 1 the equation reduces to the classical Cattaneo–Vernotte model:
    ∂²T / ∂x₃²  =  (1/λ₀) [ ∂T/∂t  +  τ_q · ∂²T / ∂t² ]

Boundary conditions:
    T(0, t) = T₀ · H(t)   (Heaviside step)
    T(H, t) = 0

Initial conditions:
    T(x₃, 0) = 0

Method
------
* Spatial:  2nd-order central finite differences on a uniform grid.
* Temporal: implicit-explicit (IMEX) scheme.
    - The diffusion term  λ₀ T_x₃x₃  is treated with Crank–Nicolson.
    - ∂T/∂t  is discretised with backward Euler.
    - ∂^β T / ∂t^β  (β = 1+γ) is approximated by the
      Grünwald–Letnikov (GL) formula (order 1), which is exact for
      polynomials and converges to the Caputo derivative when T(0) = 0.

GL weights:  w₀ = 1,  wₖ = wₖ₋₁ · (k − 1 − β) / k
GL formula:  D^β T(t_n) ≈ Δt^{−β} Σ_{k=0}^{n} wₖ T_{n−k}

Time-stepping equation (after rearranging):

    [1/Δt + C_γ Δt^{−β} w₀ − λ₀ r_CN · L₂] T^{n+1}
        = T^n / Δt  +  λ₀ r_CN · L₂ T^n
          − C_γ Δt^{−β} Σ_{k=1}^{n+1} wₖ T^{n+1−k}

where L₂ is the tridiagonal second-difference operator and r_CN = 0.5.

Returns
-------
x3_grid : ndarray (N_x,)
t_grid  : ndarray (N_t,)
T       : ndarray (N_x, N_t)  temperature field T^(1)(x₃, t)
"""

from __future__ import annotations

import numpy as np
from math import gamma as math_gamma
from scipy.linalg import solve_banded

from . import config as cfg


def _gl_weights(beta: float, n_max: int) -> np.ndarray:
    """Return Grünwald–Letnikov weights w₀ … w_{n_max} for order *beta*."""
    w = np.empty(n_max + 1)
    w[0] = 1.0
    for k in range(1, n_max + 1):
        w[k] = w[k - 1] * (k - 1.0 - beta) / k
    return w


def solve_temperature(
    N_x: int = 120,
    N_t: int = 300,
    gamma: float | None = None,
    tau_q: float | None = None,
    lam: float | None = None,
    H: float | None = None,
    T_max: float | None = None,
    T0: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the fractional CV heat equation on [0, H] × [0, T_max].

    Parameters
    ----------
    N_x, N_t : grid resolution (spatial, temporal)
    gamma     : Caputo order γ (default from config)
    tau_q     : phase-lag τ_q (default from config)
    lam       : thermal diffusivity λ₀ (default from config)
    H, T_max  : domain bounds (default from config)
    T0        : Heaviside boundary amplitude (default from config)

    Returns
    -------
    x3_grid : (N_x,)    — uniform x₃ grid
    t_grid  : (N_t,)    — uniform time grid
    T       : (N_x, N_t) — temperature field  T^(1)(x₃, t)
    """
    gamma = cfg.GAMMA if gamma is None else gamma
    tau_q = cfg.TAU_Q if tau_q is None else tau_q
    lam   = cfg.LAMBDA_0 if lam is None else lam
    H     = cfg.H if H is None else H
    T_max = cfg.T_MAX if T_max is None else T_max
    T0    = cfg.T0_BC if T0 is None else T0

    beta  = 1.0 + gamma                        # fractional order of the GL term
    C_gam = (tau_q ** gamma) / math_gamma(1.0 + gamma)

    dx = H / (N_x - 1)
    dt = T_max / (N_t - 1)

    x3 = np.linspace(0.0, H, N_x)
    t_arr = np.linspace(0.0, T_max, N_t)

    # Pre-compute GL weights up to n = N_t
    gl_w = _gl_weights(beta, N_t)

    # Coefficient of the memory term: Cγ · Δt^{-β}
    mem_coeff = C_gam * (dt ** (-beta))

    # Crank–Nicolson spatial coefficient: λ₀/(2Δx²).
    # Do NOT include Δt here — it enters separately via the 1/Δt time term.
    r = lam / (2.0 * dx ** 2)              # λ₀/(2Δx²)

    # ── Build banded tridiagonal LHS matrix (interior only) ─────────────────
    # Rows 2 … N_x-1  (interior nodes, indices 1 … N_x-2 in 0-based)
    Ni = N_x - 2   # number of interior nodes

    # LHS diagonal entry: (1/Δt + C_γ Δt^{-β} w₀) + 2·λ₀/(2Δx²)
    diag_val = (1.0 / dt + mem_coeff * gl_w[0]) + 2.0 * r
    off_val  = -r

    # scipy solve_banded:  ab[0] = super-diag, ab[1] = diag, ab[2] = sub-diag
    ab = np.zeros((3, Ni))
    ab[0, 1:] = off_val    # super-diagonal (starts at column 1)
    ab[1, :]  = diag_val   # main diagonal
    ab[2, :-1] = off_val   # sub-diagonal  (ends at column N-2)

    # ── Storage ──────────────────────────────────────────────────────────────
    T = np.zeros((N_x, N_t))
    # IC: T(x₃, 0) = 0  (already zeros)
    # BC: T(0, t>0) = T0  (Heaviside)
    T[0, 1:] = T0
    # BC: T(H, t) = 0  (already zeros)

    # ── Time stepping ─────────────────────────────────────────────────────────
    for n in range(N_t - 1):
        T_int = T[1:-1, n]          # interior values at step n
        T_bc_left  = T[0, n + 1]   # = T0 (known BC)
        T_bc_right = T[-1, n + 1]  # = 0  (known BC)

        # CN explicit part of the diffusion term (rhs contribution)
        d2T_n = np.zeros(Ni)
        d2T_n[1:-1] = r * (T_int[:-2] - 2.0 * T_int[1:-1] + T_int[2:])
        d2T_n[0]   = r * (T_bc_left  - 2.0 * T_int[0]  + T_int[1])
        d2T_n[-1]  = r * (T_int[-2]  - 2.0 * T_int[-1] + T_bc_right)

        # Memory accumulation: − mem_coeff · Σ_{k=1}^{n+1} wₖ T^{n+1−k}
        #   terms with k=1…n  → T^{n},T^{n-1},…,T^0  (all available)
        #   term  with k=n+1  → T^0 = 0 (IC)
        mem_rhs = np.zeros(Ni)
        for k in range(1, n + 2):
            idx = n + 1 - k          # time index of T^{n+1-k}
            mem_rhs -= gl_w[k] * T[1:-1, idx]
        mem_rhs *= mem_coeff

        # RHS vector
        rhs = T_int / dt + d2T_n + mem_rhs

        # Correct for BC contributions to the CN implicit part
        rhs[0]  += r * T_bc_left
        rhs[-1] += r * T_bc_right

        # Solve tridiagonal system
        T[1:-1, n + 1] = solve_banded((1, 1), ab, rhs)

    return x3, t_arr, T


def temperature_at(
    x3_query: np.ndarray,
    t_query: np.ndarray,
    x3_grid: np.ndarray,
    t_grid: np.ndarray,
    T_field: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of the pre-computed temperature field.

    Parameters
    ----------
    x3_query : 1-D array of x₃ query points
    t_query  : 1-D array of t  query points (same length)
    x3_grid, t_grid, T_field : outputs of :func:`solve_temperature`

    Returns
    -------
    T_vals : 1-D array of interpolated temperature values
    """
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (x3_grid, t_grid), T_field, method="linear", bounds_error=False,
        fill_value=0.0,
    )
    pts = np.column_stack([x3_query, t_query])
    return interp(pts)

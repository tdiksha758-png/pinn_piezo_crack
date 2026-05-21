"""Post-processing utilities: stress intensity factors (SIF) and visualisation.

Stress intensity factors
------------------------
K_Ia(t) = lim_{x₃→a⁻}  { √(2π(a−x₃)) · τ₁₁(0, x₃, t) }
K_Ib(t) = lim_{x₃→b⁺}  { √(2π(x₃−b)) · τ₁₁(0, x₃, t) }

Numerically evaluated by extrapolating  √(2π δ) · τ₁₁  to δ→0
using a sequence of points near each crack tip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")       # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from . import config as cfg
from .pde_residuals import first_order_derivatives, constitutive


# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate τ₁₁(0, x₃_arr, t) from the network
# ─────────────────────────────────────────────────────────────────────────────

def _tau11_at_left_face(
    net,
    x3_np: np.ndarray,
    t_val: float,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Evaluate τ₁₁ at x₁=0, given x₃ array and scalar time t_val."""
    N = len(x3_np)
    x1 = torch.zeros(N, 1, device=device, dtype=dtype, requires_grad=True)
    x3 = torch.tensor(x3_np.reshape(-1, 1), device=device, dtype=dtype,
                       requires_grad=True)
    t  = torch.full((N, 1), t_val, device=device, dtype=dtype)

    u1, u3, phi = net(x1, x3, t)
    d = first_order_derivatives(u1, u3, phi, x1, x3)
    c = constitutive(d)
    return c["tau11"].detach().cpu().numpy().ravel()


# ─────────────────────────────────────────────────────────────────────────────
# SIF computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_sif(
    net,
    t_values: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    n_tip: int = 30,
    delta_min: float = 1e-4,
    delta_max: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute K_Ia(t) and K_Ib(t) for each time in *t_values*.

    The limit is approximated by evaluating  f(δ) = √(2πδ)·τ₁₁  at a sequence
    of  δ ∈ [delta_min, delta_max]  near each tip and extrapolating to δ=0
    with a 1-D polynomial fit.

    Returns
    -------
    K_Ia, K_Ib : (len(t_values),)
    """
    a = cfg.A_CRACK
    b = cfg.B_CRACK

    deltas = np.linspace(delta_min, delta_max, n_tip)

    K_Ia = np.zeros(len(t_values))
    K_Ib = np.zeros(len(t_values))

    # NOTE: no torch.no_grad() here — autograd is required to evaluate τ₁₁
    # via first_order_derivatives. net.eval() only affects BatchNorm/Dropout.
    net.eval()
    for i, t_val in enumerate(t_values):
        # Near tip a (approach from below: x₃ = a − δ)
        x3_a = a - deltas
        tau_a = _tau11_at_left_face(net, x3_a, t_val, device, dtype)
        f_a   = np.sqrt(2.0 * np.pi * deltas) * tau_a
        # Linear extrapolation to δ=0
        K_Ia[i] = np.polyval(np.polyfit(deltas, f_a, 1), 0.0)

        # Near tip b (approach from above: x₃ = b + δ)
        x3_b = b + deltas
        tau_b = _tau11_at_left_face(net, x3_b, t_val, device, dtype)
        f_b   = np.sqrt(2.0 * np.pi * deltas) * tau_b
        K_Ib[i] = np.polyval(np.polyfit(deltas, f_b, 1), 0.0)

    return K_Ia, K_Ib


# ─────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_temperature(
    x3_grid: np.ndarray,
    t_grid: np.ndarray,
    T_field: np.ndarray,
    save_dir: str | Path = "figures",
) -> None:
    """Contour plot of T^(1)(x₃, t)."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    TT, XX = np.meshgrid(t_grid, x3_grid)
    cf = ax.contourf(TT, XX, T_field, levels=40, cmap="hot")
    fig.colorbar(cf, ax=ax, label="T⁽¹⁾ (K)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x₃ (m)")
    ax.set_title("Fractional CV temperature field T⁽¹⁾(x₃, t)")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "temperature_field.png", dpi=150)
    plt.close(fig)
    print("Saved temperature_field.png")


def plot_sif(
    t_values: np.ndarray,
    K_Ia: np.ndarray,
    K_Ib: np.ndarray,
    save_dir: str | Path = "figures",
) -> None:
    """Plot K_Ia(t) and K_Ib(t)."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_values, K_Ia / 1e6, label=r"$K_{Ia}$ (MPa√m)", lw=2)
    ax.plot(t_values, K_Ib / 1e6, label=r"$K_{Ib}$ (MPa√m)", lw=2, ls="--")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("SIF (MPa·√m)")
    ax.set_title("Stress Intensity Factors vs. Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "sif_vs_time.png", dpi=150)
    plt.close(fig)
    print("Saved sif_vs_time.png")


def plot_loss_history(
    history: list[dict[str, float]],
    save_dir: str | Path = "figures",
) -> None:
    """Plot training loss components over iterations."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    keys = ["total", "pde", "bc1", "bc2", "bc3", "bc4", "bc5", "far"]
    data = {k: [h[k] for h in history if k in h] for k in keys}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.semilogy(data["total"], lw=2, color="k")
    ax.set_title("Total loss")
    ax.set_xlabel("Iteration");  ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for k in ["pde", "bc1", "bc2", "bc3", "bc4", "bc5", "far"]:
        if data[k]:
            ax.semilogy(data[k], label=k, lw=1.5)
    ax.set_title("Loss components")
    ax.set_xlabel("Iteration")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(save_dir) / "loss_history.png", dpi=150)
    plt.close(fig)
    print("Saved loss_history.png")


def plot_displacement_field(
    net,
    t_val: float,
    device: torch.device,
    dtype: torch.dtype,
    nx: int = 80,
    nz: int = 80,
    save_dir: str | Path = "figures",
) -> None:
    """Contour plots of u₁, u₃, φ at a given time t_val."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    x1_lin = np.linspace(0.0, cfg.L_TRUNC, nx)
    x3_lin = np.linspace(0.0, cfg.H, nz)
    X1, X3 = np.meshgrid(x1_lin, x3_lin)

    x1_t = torch.tensor(X1.ravel().reshape(-1, 1), dtype=dtype, device=device)
    x3_t = torch.tensor(X3.ravel().reshape(-1, 1), dtype=dtype, device=device)
    t_t  = torch.full_like(x1_t, t_val)

    net.eval()
    with torch.no_grad():
        u1, u3, phi = net(x1_t, x3_t, t_t)

    fields = {
        "u₁ (m)": u1.cpu().numpy().reshape(nz, nx),
        "u₃ (m)": u3.cpu().numpy().reshape(nz, nx),
        "φ (V)":  phi.cpu().numpy().reshape(nz, nx),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, F) in zip(axes, fields.items()):
        cf = ax.contourf(X1, X3, F, levels=40, cmap="RdBu_r")
        fig.colorbar(cf, ax=ax, label=label)
        ax.set_xlabel("x₁ (m)");  ax.set_ylabel("x₃ (m)")
        ax.set_title(f"{label}  at t={t_val:.2f}s")
        # Mark crack location
        ax.axvline(x=0, color="k", lw=1, ls="--")
        ax.axhline(y=cfg.A_CRACK, color="r", lw=1, xmin=0, xmax=0.02)
        ax.axhline(y=cfg.B_CRACK, color="r", lw=1, xmin=0, xmax=0.02)

    fig.tight_layout()
    fname = f"fields_t{t_val:.2f}.png".replace(".", "p")
    fig.savefig(Path(save_dir) / fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

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
    n_tip: int = 60,
    delta_min: float = 1e-3,
    delta_max: float = 0.01,
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
        K_Ia[i] = np.polyval(np.polyfit(deltas, f_a, 2), 0.0)

        # Near tip b (approach from above: x₃ = b + δ)
        x3_b = b + deltas
        tau_b = _tau11_at_left_face(net, x3_b, t_val, device, dtype)
        f_b   = np.sqrt(2.0 * np.pi * deltas) * tau_b
        K_Ib[i] = np.polyval(np.polyfit(deltas, f_b, 2), 0.0)

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
    T_nd = T_field / cfg.T0_BC
    H = cfg.H
    F_grid = (cfg.LAMBDA_0 * t_grid)/(H**2)
    TT, XX = np.meshgrid(F_grid, x3_grid)
    cf = ax.contourf(TT, XX/H, T_nd, levels=40, cmap="hot")
    fig.colorbar(cf, ax=ax, label="T⁽¹⁾ (K)/T_0")
    ax.set_xlabel("Fourier Number F")
    ax.set_ylabel("x₃/H")
    ax.set_title(r"Normalized Temperature Field $T^{(1)}/T_0$"
)
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

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    H = cfg.H
    F_values = (cfg.LAMBDA_0 * t_values)/(H**2)

    c = cfg.B_CRACK - cfg.A_CRACK
    K_ref = cfg.KAPPA_33 * cfg.T0_BC * np.sqrt(np.pi * c)

    K_Ia_nd = K_Ia / K_ref
    K_Ib_nd = K_Ib / K_ref

    # ---------------- K_Ia plot ----------------
    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(F_values, K_Ia_nd, lw=2)

    ax.set_xlabel("Fourier Number F")
    ax.set_ylabel(r"$K_{Ia}/(k_{33}T_0\sqrt{\pi c})$")
    ax.set_title(r"Stress Intensity Factor $K_{Ia}$")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        Path(save_dir) / "K_Ia_vs_time.png",
        dpi=150
    )

    plt.close(fig)

    print("Saved K_Ia_vs_time.png")


    # ---------------- K_Ib plot ----------------
    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(F_values, K_Ib_nd, lw=2)

    ax.set_xlabel("Fourier Number F")
    ax.set_ylabel(r"$K_{Ib}/(k_{33}T_0\sqrt{\pi c})$")
    ax.set_title(r"Stress Intensity Factor $K_{Ib}$")

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        Path(save_dir) / "K_Ib_vs_time.png",
        dpi=150
    )

    plt.close(fig)

    print("Saved K_Ib_vs_time.png")


    
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



def plot_tau0_field(
    x3_grid: np.ndarray,
    t_grid: np.ndarray,
    tau0_field: np.ndarray,
    save_dir: str | Path = "figures",
) -> None:
    """Contour plot of τ₀(x₃,F)."""

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    # Non-dimensional thermal loading
    tau0_nd = tau0_field / (cfg.KAPPA_11 * cfg.T0_BC)


    H = cfg.H

    # Fourier number
    F_grid = (cfg.LAMBDA_0 * t_grid) / (H ** 2)

    # Meshgrid
    FF, XX = np.meshgrid(F_grid, x3_grid)

    # Contour plot
    cf = ax.contourf(
        FF,
        XX/H,
        tau0_nd,
        levels=50,
        cmap="RdBu_r"
    )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\tau_0/(k_{11}T_0)$")

    ax.set_xlabel("Fourier Number F")
    ax.set_ylabel(r"$x_3/H$")
    ax.set_title(
    r"Normalized Thermal Loading $\tau_0/(k_{11}T_0)$"
)

    fig.tight_layout()

    fig.savefig(
        Path(save_dir) / "tau0_field.png",
        dpi=150
    )

    plt.close(fig)

    print("Saved tau0_field.png")

def plot_tau11_near_tip(
    net,
    t_val,
    device,
    dtype,
    tip="a",
    n_points=200,
    delta_min=1e-4,
    delta_max=0.05,
    save_dir="figures",
):
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if tip == "a":
        x3_vals = cfg.A_CRACK - np.linspace(delta_min, delta_max, n_points)
        r_vals = cfg.A_CRACK - x3_vals
    else:
        x3_vals = cfg.B_CRACK + np.linspace(delta_min, delta_max, n_points)
        r_vals = x3_vals - cfg.B_CRACK

    tau11 = _tau11_at_left_face(
        net,
        x3_vals,
        t_val,
        device,
        dtype,
    )

    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(r_vals, tau11, lw=2)

    ax.set_xlabel("Distance from crack tip r")
    ax.set_ylabel(r"$\tau_{11}$")
    ax.set_title(f"Near-tip stress field at tip {tip}")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        Path(save_dir) / f"tau11_tip_{tip}.png",
        dpi=150,
    )

    plt.close(fig)

    print(f"Saved tau11_tip_{tip}.png")
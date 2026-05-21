"""Entry point for the PZT-4 piezoelectric PINN solver.

Workflow
--------
1. Solve the 1-D fractional Cattaneo–Vernotte heat equation numerically
   → temperature field T^(1)(x₃, t)

2. Compute the thermal-loading function τ₀(x₃, t) and the force/moment
   balancing coefficients A(t), B(t).

3. Train the mechanics PINN (u₁, u₃, φ) subject to the 3 governing PDEs
   and boundary conditions BC1–BC5.

4. Post-process: compute stress intensity factors K_Ia(t), K_Ib(t) and
   generate plots.

Usage::
    uv run python main.py
    uv run python main.py --gamma 0.8 --no-lbfgs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import src.config as cfg
from src.temperature import solve_temperature
from src.thermal_loading import compute_AB, compute_tau0_field, make_tau0_interpolator
from src.network import MechanicsNet
from src.trainer import Trainer
from src.postprocess import (
    plot_temperature,
    plot_sif,
    plot_loss_history,
    plot_displacement_field,
    compute_sif,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PZT-4 piezoelectric PINN solver")
    p.add_argument("--gamma",      type=float, default=cfg.GAMMA,
                   help="Caputo fractional order γ  (0 < γ ≤ 1, default %(default)s)")
    p.add_argument("--adam-iter",  type=int,   default=cfg.MAX_ITER_ADAM,
                   help="Adam iterations (default %(default)s)")
    p.add_argument("--lbfgs-iter", type=int,   default=cfg.MAX_ITER_LBFGS,
                   help="L-BFGS iterations (default %(default)s)")
    p.add_argument("--no-lbfgs",   action="store_true",
                   help="Skip L-BFGS phase")
    p.add_argument("--nx-temp",    type=int,   default=120,
                   help="Temperature solver: spatial grid points")
    p.add_argument("--nt-temp",    type=int,   default=300,
                   help="Temperature solver: time-grid points")
    p.add_argument("--checkpoint", type=str,   default="checkpoints/model.pt",
                   help="Path to save/load the model checkpoint")
    p.add_argument("--load",       action="store_true",
                   help="Load checkpoint and skip training (post-process only)")
    p.add_argument("--figures",    type=str,   default="figures",
                   help="Directory for output figures")
    p.add_argument("--n-int",      type=int,   default=cfg.N_INTERIOR,
                   help="Interior collocation points (default %(default)s)")
    p.add_argument("--n-bc",       type=int,   default=cfg.N_BOUNDARY,
                   help="Boundary points per segment (default %(default)s)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float64
    print(f"Running on {device}  |  γ = {args.gamma}")

    # ── Step 1: Temperature ───────────────────────────────────────────────────
    print("\n[1/4] Solving fractional heat equation …")
    x3_grid, t_grid, T_field = solve_temperature(
        N_x=args.nx_temp,
        N_t=args.nt_temp,
        gamma=args.gamma,
    )
    plot_temperature(x3_grid, t_grid, T_field, save_dir=args.figures)

    # ── Step 2: Thermal loading ───────────────────────────────────────────────
    print("[2/4] Computing τ₀(x₃, t), A(t), B(t) …")
    A, B          = compute_AB(x3_grid, T_field)
    tau0_field    = compute_tau0_field(x3_grid, t_grid, T_field, A, B)
    tau0_fn       = make_tau0_interpolator(x3_grid, t_grid, tau0_field)

    # ── Step 3: PINN training ─────────────────────────────────────────────────
    print("[3/4] Constructing PINN …")
    net = MechanicsNet(cfg.MECH_LAYERS)

    trainer = Trainer(
        net, tau0_fn,
        device=device, dtype=dtype,
        n_interior=args.n_int,
        n_boundary=args.n_bc,
        max_iter_adam=args.adam_iter,
        max_iter_lbfgs=0 if args.no_lbfgs else args.lbfgs_iter,  # type: ignore[arg-type]
    )

    if args.load and Path(args.checkpoint).exists():
        trainer.load(args.checkpoint)
    else:
        trainer.train()
        trainer.save(args.checkpoint)

    plot_loss_history(trainer.history, save_dir=args.figures)

    # ── Step 4: Post-processing ───────────────────────────────────────────────
    print("[4/4] Post-processing …")
    t_plot = np.array([0.2, 0.5, 1.0, 1.5, 2.0])
    t_plot = t_plot[t_plot <= cfg.T_MAX]

    K_Ia, K_Ib = compute_sif(net, t_plot, device, dtype)
    plot_sif(t_plot, K_Ia, K_Ib, save_dir=args.figures)

    for tv in t_plot[:3]:                  # plot fields for first 3 snapshots
        plot_displacement_field(net, tv, device, dtype, save_dir=args.figures)

    print("\nDone.  Figures saved to:", args.figures)
    print(f"  K_Ia at t={t_plot[-1]:.1f}s : {K_Ia[-1]/1e6:.4f} MPa√m")
    print(f"  K_Ib at t={t_plot[-1]:.1f}s : {K_Ib[-1]/1e6:.4f} MPa√m")


if __name__ == "__main__":
    main()

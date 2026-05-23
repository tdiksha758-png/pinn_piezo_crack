"""Entry point for the Triclinic Piezoelectric Interface PINN solver.

Workflow
--------
1. Instantiate the 7-mode MultiModeNet.
2. Train with Adam (+ optional L-BFGS) using ACTIVE_PROBLEM.
3. Save checkpoint and plot loss history.

Usage::
    uv run python main.py
    uv run python main.py --adam-iter 20000 --no-lbfgs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import src.config as cfg
from src.problem_definition import ACTIVE_PROBLEM, MultiModeNet
from src.trainer import Trainer
from src.postprocess import plot_loss_history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Triclinic piezoelectric interface PINN solver"
    )
    p.add_argument("--adam-iter",  type=int,   default=cfg.MAX_ITER_ADAM,
                   help="Adam iterations (default %(default)s)")
    p.add_argument("--lbfgs-iter", type=int,   default=cfg.MAX_ITER_LBFGS,
                   help="L-BFGS iterations (default %(default)s)")
    p.add_argument("--no-lbfgs",   action="store_true",
                   help="Skip L-BFGS phase")
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
    print(f"Problem : {ACTIVE_PROBLEM.name}")
    print(f"Running on {device}  |  dtype: {dtype}")

    # ── Instantiate network ───────────────────────────────────────────────────
    net = MultiModeNet().to(device=device, dtype=dtype)

    trainer = Trainer(
        net,
        device=device, dtype=dtype,
        n_interior=args.n_int,
        n_boundary=args.n_bc,
        max_iter_adam=args.adam_iter,
        max_iter_lbfgs=0 if args.no_lbfgs else args.lbfgs_iter,
    )

    # ── Train or load checkpoint ──────────────────────────────────────────────
    if args.load and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["model_state"])
        trainer.history = ckpt.get("history", [])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        trainer.train()
        trainer.save(args.checkpoint)

    # ── Plot loss history ─────────────────────────────────────────────────────
    if trainer.history:
        plot_loss_history(trainer.history, save_dir=args.figures)

    print("\nDone.  Figures saved to:", args.figures)


if __name__ == "__main__":
    main()

"""Training loop for the piezoelectric PINN.

Phase 1 — Adam optimiser  (fast global exploration)
Phase 2 — L-BFGS          (high-precision fine-tuning)

Usage::
    from src.trainer import Trainer
    trainer = Trainer(net, tau0_fn)
    trainer.train()
    trainer.save("checkpoints/model.pt")
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import math
from torch import Tensor

from . import config as cfg
from .loss import pinn_loss
from .network import MechanicsNet


class Trainer:
    """Manages Adam + L-BFGS training for MechanicsNet."""

    def __init__(
        self,
        net: MechanicsNet,
        tau0_fn,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        n_interior: int = cfg.N_INTERIOR,
        n_boundary: int = cfg.N_BOUNDARY,
        lr_adam: float = cfg.LR_ADAM,
        max_iter_adam: int = cfg.MAX_ITER_ADAM,
        max_iter_lbfgs: int = cfg.MAX_ITER_LBFGS,
        log_every: int = 10,
        w_pde: float = cfg.W_PDE,
        w_bc:  float = cfg.W_BC,
        
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype  = dtype
        self.net    = net.to(device=device, dtype=dtype)
        self.tau0_fn = tau0_fn

        self.n_int = n_interior
        self.n_bc  = n_boundary

        self.lr_adam      = lr_adam
        self.max_adam     = max_iter_adam
        self.max_lbfgs    = max_iter_lbfgs
        self.log_every    = log_every

        self.w_pde = w_pde
        self.w_bc  = w_bc
        # self.w_far = w_far

        self.history: list[dict[str, float]] = []

    # ── internal loss wrapper ─────────────────────────────────────────────────
    def _loss(self) -> tuple[Tensor, dict[str, Tensor]]:
        return pinn_loss(
            self.net, self.tau0_fn,
            self.n_int, self.n_bc,
            self.device, self.dtype,
            w_pde=self.w_pde, w_bc=self.w_bc, 
        )

    # ── Phase 1: Adam ─────────────────────────────────────────────────────────
    def _train_adam(self) -> None:
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr_adam)
        t0 = time.perf_counter()

        for step in range(1, self.max_adam + 1):
            optimiser.zero_grad()
            loss, comps = self._loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
            self.net.parameters(),
            max_norm=1.0
        )

            # --- Diagnostics: gradient & parameter norms ---
            grad_norm_sq = 0.0
            param_norm_sq = 0.0
            for p in self.net.parameters():
                if p.grad is not None:
                    try:
                        gnorm = float(p.grad.detach().norm().item())
                    except Exception:
                        gnorm = float(torch.norm(p.grad.detach()).cpu().item())
                    grad_norm_sq += gnorm * gnorm
                try:
                    pnorm = float(p.detach().norm().item())
                except Exception:
                    pnorm = float(torch.norm(p.detach()).cpu().item())
                param_norm_sq += pnorm * pnorm

            grad_norm = math.sqrt(grad_norm_sq)
            param_norm = math.sqrt(param_norm_sq)

            # Early detect non-finite / exploding gradients
            if not math.isfinite(grad_norm) or not math.isfinite(float(loss)):
                print(f"Non-finite detected at step {step}: loss={float(loss)}, grad_norm={grad_norm}")
                return

            optimiser.step()

            row = {k: float(v) for k, v in comps.items()}
            row["grad_norm"] = grad_norm
            row["param_norm"] = param_norm
            self.history.append(row)

            if step % self.log_every == 0:

                elapsed = time.perf_counter() - t0

                print(
                    f"\nEpoch : {step}/{self.max_adam}"
                    f"\nTotal Loss : {row['total']:.6e}"

                    f"\n\n--- PDE ---"
                    f"\nPDE Loss   : {row['pde']:.6e}"

                    f"\n\n--- Boundary Losses ---"
                    f"\nBC1 : {row['bc1']:.6e}"
                    f"\nBC2 : {row['bc2']:.6e}"
                    f"\nBC3 : {row['bc3']:.6e}"
                    f"\nBC4 : {row['bc4']:.6e}"
                    f"\nBC5 : {row['bc5']:.6e}"

                    f"\n\nBC Total   : "
                    f"{(row['bc1']+row['bc2']+row['bc3']+row['bc4']+row['bc5']):.6e}"

                    f"\n\n--- Other ---"
                    f"\nGrad Norm  : {row['grad_norm']:.6e}"
                    f"\nParam Norm : {row['param_norm']:.6e}"
                    f"\nElapsed    : {elapsed:.1f} s\n"
                )

    # ── Phase 2: L-BFGS ──────────────────────────────────────────────────────
    def _train_lbfgs(self) -> None:
        optimiser = torch.optim.LBFGS(
            self.net.parameters(),
            max_iter=self.max_lbfgs,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        step_counter = [0]
        t0 = time.perf_counter()

        def closure():
            optimiser.zero_grad()
            loss, comps = self._loss()
            loss.backward()
            step_counter[0] += 1
            if step_counter[0] % self.log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"LBFGS {step_counter[0]:5d}  "
                    f"loss={float(comps['total']):.3e}  "
                    f"[{elapsed:.1f}s]"
                )
            return loss

        optimiser.step(closure)

    # ── Public API ────────────────────────────────────────────────────────────
    def train(self) -> None:
        """Run Adam then (optionally) L-BFGS."""
        print(f"Device: {self.device}  |  dtype: {self.dtype}")
        print(f"Parameters: {sum(p.numel() for p in self.net.parameters()):,}")
        print("=" * 60)
        print("Phase 1 — Adam")
        self._train_adam()
        if self.max_lbfgs > 0:
            print("Phase 2 — L-BFGS")
            self._train_lbfgs()
        else:
            print("Phase 2 — L-BFGS skipped (max_iter_lbfgs=0)")
        print("Training complete.")

    def save(self, path: str | Path) -> None:
        """Save model weights and training history to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.net.state_dict(), "history": self.history},
            path,
        )
        print(f"Checkpoint saved → {path}")

    def load(self, path: str | Path) -> None:
        """Load model weights from *path*."""
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        print(f"Checkpoint loaded ← {path}")
    

    
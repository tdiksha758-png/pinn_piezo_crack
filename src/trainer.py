"""Training loop for the piezoelectric PINN.

Phase 1 — Adam optimiser  (fast global exploration)
Phase 2 — L-BFGS          (high-precision fine-tuning)

Usage::
    from src.trainer import Trainer
    from src.problem_definition import ACTIVE_PROBLEM

    # (optional) inject runtime params before training:
    ACTIVE_PROBLEM.set_param("tau0_fn", tau0_fn)

    trainer = Trainer(net)
    trainer.train()
    trainer.save("checkpoints/model.pt")
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import Tensor

from . import config as cfg
from .loss import pinn_loss


class Trainer:
    """Manages Adam + L-BFGS training for the PINN."""

    def __init__(
        self,
        net,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        n_interior: int = cfg.N_INTERIOR,
        n_boundary: int = cfg.N_BOUNDARY,
        n_ic: int = cfg.N_IC,
        lr_adam: float = cfg.LR_ADAM,
        max_iter_adam: int = cfg.MAX_ITER_ADAM,
        max_iter_lbfgs: int = cfg.MAX_ITER_LBFGS,
        log_every: int = 500,
        w_pde: float = cfg.W_PDE,
        w_ic:  float = cfg.W_IC,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype  = dtype
        self.net    = net.to(device=device, dtype=dtype)

        self.n_int = n_interior
        self.n_bc  = n_boundary
        self.n_ic  = n_ic

        self.lr_adam      = lr_adam
        self.max_adam     = max_iter_adam
        self.max_lbfgs    = max_iter_lbfgs
        self.log_every    = log_every

        self.w_pde = w_pde
        self.w_ic  = w_ic

        self.history: list[dict[str, float]] = []

    # ── internal loss wrapper ─────────────────────────────────────────────────
    def _loss(self) -> tuple[Tensor, dict[str, Tensor]]:
        return pinn_loss(
            self.net,
            N_int=self.n_int,
            N_bc=self.n_bc,
            N_ic=self.n_ic,
            device=self.device,
            dtype=self.dtype,
            w_pde=self.w_pde,
            w_ic=self.w_ic,
        )

    # ── Phase 1: Adam ─────────────────────────────────────────────────────────
    def _train_adam(self) -> None:
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr_adam)
        t0 = time.perf_counter()

        for step in range(1, self.max_adam + 1):
            optimiser.zero_grad()
            loss, comps = self._loss()
            loss.backward()
            optimiser.step()

            row = {k: float(v) for k, v in comps.items()}
            self.history.append(row)

            if step % self.log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"Adam {step:6d}/{self.max_adam}  "
                    f"loss={row['total']:.3e}  "
                    f"pde={row['pde']:.3e}  "
                    f"bc={row['bc_total']:.3e}  "
                    f"ic={row['ic_total']:.3e}  "
                    f"[{elapsed:.1f}s]"
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

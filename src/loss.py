"""Weighted total PINN loss — driven by problem_definition.ACTIVE_PROBLEM.

Loss = w_pde · Σ‖Rᵢ‖²                    (PDE residuals)
     + Σ bc.weight · ‖BCⱼ‖²              (boundary conditions, per-BC weights)
     + w_ic  · Σ ic.weight · ‖ICₖ‖²     (initial conditions, optional)

The governing equations, boundary conditions, and initial conditions are
read entirely from  src/problem_definition.py  — no other file needs to
change when switching problems.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import config as cfg


def _mse(r: Tensor) -> Tensor:
    return (r ** 2).mean()


def pinn_loss(
    net,
    N_int: int = cfg.N_INTERIOR,
    N_bc:  int = cfg.N_BOUNDARY,
    N_ic:  int = cfg.N_IC,
    device: torch.device | None = None,
    dtype:  torch.dtype = torch.float64,
    w_pde:  float = cfg.W_PDE,
    w_ic:   float = cfg.W_IC,
    problem=None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute total PINN loss and return component dictionary.

    Parameters
    ----------
    net     : network (callable)
    N_int   : number of interior (PDE) collocation points
    N_bc    : number of points per boundary segment
    N_ic    : number of initial-condition collocation points
    device, dtype : torch device / dtype
    w_pde   : global PDE loss weight
    w_ic    : global IC loss weight (ignored when problem has no ICs)
    problem : ProblemSpec from problem_definition.py;
              defaults to ACTIVE_PROBLEM when None

    Returns
    -------
    total_loss : scalar Tensor (with grad)
    components : dict[str, Tensor] — individual terms (detached) for logging
    """
    from .problem_definition import ACTIVE_PROBLEM
    if problem is None:
        problem = ACTIVE_PROBLEM

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dom = problem.domain
    x1_lo, x1_hi = dom.x1_range
    x3_lo, x3_hi = dom.x3_range
    t_lo,  t_hi  = dom.t_range if dom.t_range is not None else (0.0, 1.0)

    # ── Interior PDE collocation points ──────────────────────────────────────
    def _sample(lo, hi, N):
        return (lo + (hi - lo) * torch.rand(N, 1, device=device, dtype=dtype)
                ).requires_grad_(True)

    x1_int = _sample(x1_lo, x1_hi, N_int)
    x3_int = _sample(x3_lo, x3_hi, N_int)
    t_int  = torch.rand(N_int, 1, device=device, dtype=dtype) * (t_hi - t_lo) + t_lo

    if dom.y_range is not None:                          # 3-D problem
        y_lo, y_hi = dom.y_range
        y_int = _sample(y_lo, y_hi, N_int)
        pde_residuals = problem.pde_fn(net, x1_int, y_int, x3_int, t_int)
    else:                                                # 2-D problem (default)
        pde_residuals = problem.pde_fn(net, x1_int, x3_int, t_int)
    loss_pde = sum(_mse(r) for r in pde_residuals)

    # ── Boundary conditions ───────────────────────────────────────────────────
    components: dict[str, Tensor] = {}
    loss_bc_total = torch.zeros(1, device=device, dtype=dtype).squeeze()

    for bc in problem.boundary_conditions:
        r = bc.residual_fn(net, N_bc, device, dtype, **problem.params)
        if isinstance(r, (tuple, list)):
            bc_loss = bc.weight * sum(_mse(ri) for ri in r)
        else:
            bc_loss = bc.weight * _mse(r)
        components[bc.name] = bc_loss.detach()
        loss_bc_total = loss_bc_total + bc_loss

    components["bc_total"] = loss_bc_total.detach()

    # ── Initial conditions (optional) ─────────────────────────────────────────
    loss_ic_total = torch.zeros(1, device=device, dtype=dtype).squeeze()
    if problem.has_initial_conditions:
        for ic in problem.initial_conditions:
            r = ic.residual_fn(net, N_ic, device, dtype, **problem.params)
            ic_loss = ic.weight * _mse(r)
            components[f"ic_{ic.name}"] = ic_loss.detach()
            loss_ic_total = loss_ic_total + ic_loss
    components["ic_total"] = loss_ic_total.detach()

    # ── Total ─────────────────────────────────────────────────────────────────
    total = w_pde * loss_pde + loss_bc_total + w_ic * loss_ic_total
    components["pde"]   = loss_pde.detach() if isinstance(loss_pde, Tensor) else torch.tensor(loss_pde)
    components["total"] = total.detach()

    return total, components

"""Automated correctness checks for the single-file problem-definition system."""
import sys, torch, numpy as np

from src.problem_definition import (
    ACTIVE_PROBLEM, BoundaryCondition, InitialCondition, DomainSpec, ProblemSpec,
    MultiModeNet,
)
from src.loss import pinn_loss

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(cond, msg):
    tag = PASS if cond else FAIL
    print(f"  {tag}  {msg}")
    if not cond:
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
print("=== Test 1: ACTIVE_PROBLEM structure ===")
check("Triclinic" in ACTIVE_PROBLEM.name, "name contains 'Triclinic'")
check(len(ACTIVE_PROBLEM.boundary_conditions) == 6, "6 BCs registered")
check(ACTIVE_PROBLEM.boundary_conditions[0].weight == 10.0, "BC1 weight == 10.0")
check(ACTIVE_PROBLEM.boundary_conditions[-1].weight == 10.0, "bc_phi_sum weight == 10.0")
check(not ACTIVE_PROBLEM.has_initial_conditions, "no ICs for this problem")
check(ACTIVE_PROBLEM.is_time_dependent, "time-dependent (t_range set)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 2: pinn_loss forward + backward (Triclinic, MultiModeNet) ===")

device = torch.device("cpu")
dtype  = torch.float64
net = MultiModeNet().to(device=device, dtype=dtype)

total, comps = pinn_loss(net, N_int=60, N_bc=20, N_ic=0,
                         device=device, dtype=dtype,
                         w_pde=1.0, w_ic=0.0,
                         problem=ACTIVE_PROBLEM)

check(total.requires_grad, "total has grad")
check("pde"      in comps, "'pde' key in components")
check("bc_total" in comps, "'bc_total' key in components")
check("ic_total" in comps, "'ic_total' key in components")
check("total"    in comps, "'total' key in components")
check(all(n in comps for n in
          ["bc_u1_sum", "bc_u2_sum", "bc_u3_sum",
           "bc_T31_sum", "bc_T32_sum", "bc_phi_sum"]),
      "per-BC keys present")
check(float(comps["ic_total"]) == 0.0, "ic_total == 0 (no ICs)")
check(float(comps["bc_total"]) > 0,    "bc_total > 0")
check(float(comps["pde"]) > 0,         "pde loss > 0")

# verify backward()
total.backward()
grads_ok = all(p.grad is not None for p in net.parameters())
check(grads_ok, "backward() produces gradients for all params")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 3: Problem WITH InitialCondition ===")

import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )
    def forward(self, x1, x3, t=None):
        return self.net(torch.cat([x1, x3], dim=-1))

def _g(out, inp):
    return torch.autograd.grad(out, inp, torch.ones_like(out),
                               create_graph=True, retain_graph=True)[0]

def simple_pde(net, x1, x3, t):
    u = net(x1, x3)
    return [_g(_g(u, x1), x1) + _g(_g(u, x3), x3)]

def simple_bc(net, N, device, dtype, **_):
    x1 = torch.zeros(N, 1, device=device, dtype=dtype)
    x3 = torch.rand (N, 1, device=device, dtype=dtype)
    return net(x1, x3)

def simple_ic(net, N, device, dtype, **_):
    x1 = torch.rand(N, 1, device=device, dtype=dtype)
    x3 = torch.rand(N, 1, device=device, dtype=dtype)
    return net(x1, x3) - torch.sin(x1)

prob_ic = ProblemSpec(
    name="Test with IC",
    domain=DomainSpec(x1_range=(0.,1.), x3_range=(0.,1.), t_range=(0.,1.)),
    pde_fn=simple_pde,
    boundary_conditions=[BoundaryCondition("bc_left", simple_bc, weight=5.0)],
    initial_conditions=[InitialCondition("ic_t0",  simple_ic, weight=8.0)],
)

check(prob_ic.has_initial_conditions, "has_initial_conditions == True")
check(prob_ic.is_time_dependent,      "is_time_dependent == True")

dtype32 = torch.float32
snet = SimpleNet().to(dtype=dtype32)
total2, comps2 = pinn_loss(snet, N_int=40, N_bc=20, N_ic=20,
                            device=device, dtype=dtype32,
                            w_pde=1.0, w_ic=1.0, problem=prob_ic)

check("ic_ic_t0" in comps2, "per-IC key 'ic_ic_t0' in components")
check(float(comps2["ic_total"]) > 0, "ic_total > 0 when IC registered")
check(float(comps2["bc_total"]) > 0, "bc_total > 0")
check(total2.requires_grad,           "total2 has grad")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 4: Steady problem (initial_conditions=None, t_range=None) ===")

prob_steady = ProblemSpec(
    name="Steady",
    domain=DomainSpec(x1_range=(0.,1.), x3_range=(0.,1.), t_range=None),
    pde_fn=simple_pde,
    boundary_conditions=[BoundaryCondition("bc_left", simple_bc, weight=1.0)],
    initial_conditions=None,
)

check(not prob_steady.has_initial_conditions, "has_initial_conditions == False")
check(not prob_steady.is_time_dependent,      "is_time_dependent == False")

total3, comps3 = pinn_loss(snet, N_int=40, N_bc=20, N_ic=0,
                            device=device, dtype=dtype32, problem=prob_steady)
check(float(comps3["ic_total"]) == 0.0, "ic_total == 0.0 for steady problem")

# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 5: postprocess.plot_loss_history with new keys ===")
from src.postprocess import plot_loss_history
import io, unittest.mock as mock

fake_history = [
    {"total": 1.0, "pde": 0.5, "bc_total": 0.4, "ic_total": 0.0,
     "bc1_crack_stress": 0.1, "bc2_disp": 0.1},
    {"total": 0.8, "pde": 0.3, "bc_total": 0.3, "ic_total": 0.0,
     "bc1_crack_stress": 0.08, "bc2_disp": 0.09},
]
with mock.patch("matplotlib.pyplot.savefig"), \
     mock.patch("matplotlib.figure.Figure.savefig"), \
     mock.patch("builtins.print"):
    try:
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            plot_loss_history(fake_history, save_dir=tmp)
        check(True, "plot_loss_history runs without error on new key schema")
    except Exception as e:
        check(False, f"plot_loss_history failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("ALL CHECKS PASSED")

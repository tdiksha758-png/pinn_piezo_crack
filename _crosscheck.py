"""Cross-check the _MULTIMODE_PROBLEM in problem_definition.py."""
import torch
import src.problem_definition as pd
from src.loss import pinn_loss
from src.problem_definition import MultiModeNet

PASS = "[PASS]"; FAIL = "[FAIL]"
dev, dt = torch.device("cpu"), torch.float64
torch.manual_seed(42)

all_ok = True
def check(label, condition):
    global all_ok
    status = PASS if condition else FAIL
    if not condition:
        all_ok = False
    print(f"  {status}  {label}")


# ── CHECK 1: _MULTIMODE_PROBLEM exists and is ACTIVE_PROBLEM ─────────────────
print("\n=== CHECK 1: Problem definition ===")
check("_MULTIMODE_PROBLEM defined",           hasattr(pd, "_MULTIMODE_PROBLEM"))
check("ACTIVE_PROBLEM is _MULTIMODE_PROBLEM", pd.ACTIVE_PROBLEM is pd._MULTIMODE_PROBLEM)
check("Problem name contains 'Triclinic'",    "Triclinic" in pd._MULTIMODE_PROBLEM.name)


# ── CHECK 2: PDE residual count ───────────────────────────────────────────────
print("\n=== CHECK 2: PDE residuals (28 expected) ===")
x = torch.rand(5, 1, requires_grad=True, dtype=dt)
z = torch.rand(5, 1, requires_grad=True, dtype=dt)
t = torch.rand(5, 1, requires_grad=True, dtype=dt)
net = MultiModeNet([3, 8, 4]).to(dtype=dt)
res = pd._MULTIMODE_PROBLEM.pde_fn(net, x, z, t)
check(f"PDE residual count = {len(res)} (expected 28)", len(res) == 28)


# ── CHECK 3: BC count, names, and weights ─────────────────────────────────────
print("\n=== CHECK 3: Boundary conditions ===")
expected_bc_names = ["bc_u1_sum", "bc_u2_sum", "bc_u3_sum",
                     "bc_T31_sum", "bc_T32_sum", "bc_phi_sum"]
actual_bcs = pd._MULTIMODE_PROBLEM.boundary_conditions
actual_names = [bc.name for bc in actual_bcs]
check(f"BC count = {len(actual_names)} (expected 6)", len(actual_names) == 6)
check("BC names match",              actual_names == expected_bc_names)
check("All BC weights == 10.0",      all(bc.weight == 10.0 for bc in actual_bcs))


# ── CHECK 4: Initial conditions and time-dependence ──────────────────────────
print("\n=== CHECK 4: ICs and time-dependence ===")
check("has_initial_conditions == False", not pd._MULTIMODE_PROBLEM.has_initial_conditions)
check("is_time_dependent == True",        pd._MULTIMODE_PROBLEM.is_time_dependent)


# ── CHECK 5: Full forward + backward pass ─────────────────────────────────────
print("\n=== CHECK 5: Forward + backward pass ===")
net2 = MultiModeNet([3, 16, 16, 4]).to(dtype=dt)
total, comps = pinn_loss(net2, N_int=80, N_bc=40, N_ic=0,
    device=dev, dtype=dt, w_pde=1.0, w_ic=0.0, problem=pd._MULTIMODE_PROBLEM)
total.backward()
check("total loss > 0 and finite",  total.item() > 0 and torch.isfinite(total))
check("pde loss > 0",               comps["pde"].item() > 0)
check("bc_total > 0",               comps["bc_total"].item() > 0)
check("ic_total == 0.0",            comps["ic_total"].item() == 0.0)
check("all gradients computed",     all(p.grad is not None for p in net2.parameters()))
for bc_key in expected_bc_names:
    check(f"{bc_key} in comps",     bc_key in comps)


# ── CHECK 6: Network parameter count (7 × single sub-net) ────────────────────
print("\n=== CHECK 6: Network parameter count ===")
from src.problem_definition import TriclinicNet
single_params = sum(p.numel() for p in TriclinicNet([3, 32, 32, 4]).parameters())
multi_params  = sum(p.numel() for p in MultiModeNet([3, 32, 32, 4]).parameters())
check(f"MultiModeNet params == 7 × TriclinicNet  ({multi_params} == 7 × {single_params})",
      multi_params == 7 * single_params)


# ── FINAL RESULT ──────────────────────────────────────────────────────────────
print()
print("=" * 50)
if all_ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED — see [FAIL] lines above")
print("=" * 50)

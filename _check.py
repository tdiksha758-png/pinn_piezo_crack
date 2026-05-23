"""Quick sanity check for the triclinic interface PINN setup."""
import torch
from src.problem_definition import MultiModeNet, ACTIVE_PROBLEM, _multimode_pde_fn
from src.loss import pinn_loss

device = torch.device('cpu')
dtype  = torch.float64
torch.manual_seed(0)

net = MultiModeNet().to(device=device, dtype=dtype)

# ── 1. Forward pass shape check ───────────────────────────────────────────────
N = 10
x = (torch.rand(N, 1, dtype=dtype) * 10.0).requires_grad_(True)
z = (torch.rand(N, 1, dtype=dtype) *  5.0).requires_grad_(True)
t =  torch.rand(N, 1, dtype=dtype).requires_grad_(True)

outs = net.nets[0](x, z, t)
assert len(outs) == 4, f"Expected 4 outputs, got {len(outs)}"
for fi, name in enumerate(["u1", "u2", "u3", "phi"]):
    assert outs[fi].shape == (N, 1), f"Wrong shape for {name}: {outs[fi].shape}"
print('u1 shape:', outs[0].shape,   '   expected (10,1)')
print('u3 shape:', outs[2].shape,   '   expected (10,1)')
print('phi shape:', outs[3].shape,  '   expected (10,1)')

# ── 2. PDE residual count ─────────────────────────────────────────────────────
res = _multimode_pde_fn(net, x, z, t)
assert len(res) == 28, f"Expected 28 PDE residuals, got {len(res)}"
print(f'PDE residuals: {len(res)}   expected 28')
assert res[0].shape == (N, 1), f"Wrong residual shape: {res[0].shape}"
print(f'Residual[0] shape: {res[0].shape}   expected (10,1)')

# ── 3. pinn_loss forward ──────────────────────────────────────────────────────
loss, comps = pinn_loss(
    net, N_int=60, N_bc=25, N_ic=0,
    device=device, dtype=dtype,
    w_pde=1.0, w_ic=0.0,
    problem=ACTIVE_PROBLEM,
)
assert loss.requires_grad,   "Loss must have gradient"
assert "pde"      in comps,  "'pde' not in components"
assert "bc_total" in comps,  "'bc_total' not in components"
assert "total"    in comps,  "'total' not in components"
assert "ic_total" in comps,  "'ic_total' not in components"
print(f'Loss={float(loss.detach()):.3e}  components: {list(comps.keys())}')

# ── 4. Backward pass ──────────────────────────────────────────────────────────
loss.backward()
grads_ok = all(p.grad is not None for p in net.parameters())
assert grads_ok, "Some parameters have no gradient after backward()"

print('=== ALL SHAPE CHECKS PASSED ===')

import sys
import os
import torch
import numpy as np

# Ensure project root on PYTHONPATH so `import src` works when running this script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import src.config as cfg
from src.network import MechanicsNet
from src.boundary_conditions import sample_crack_face
from src.pde_residuals import first_order_derivatives, constitutive
from src.temperature import solve_temperature
from src.thermal_loading import compute_AB, compute_tau0_field, make_tau0_interpolator

# Recreate small thermal field
x3_grid, t_grid, T_field = solve_temperature(N_x=60, N_t=60, gamma=cfg.GAMMA)
A, B = compute_AB(x3_grid, T_field)
tau0_field = compute_tau0_field(x3_grid, t_grid, T_field, A, B)
tau0_fn = make_tau0_interpolator(x3_grid, t_grid, tau0_field)

# Load trained model
ck = torch.load('checkpoints/model.pt', map_location='cpu')
net = MechanicsNet(cfg.MECH_LAYERS)
net.load_state_dict(ck['model_state'])
net.eval()

dtype = torch.float64
device = torch.device('cpu')

# Sample crack-face points
N = 2000
x1, x3, t = sample_crack_face(N, device=device, dtype=dtype)
# Ensure grads for derivatives
# Convert sample tensors to network parameter dtype to avoid dtype mismatch
param_dtype = next(net.parameters()).dtype
if x1.dtype != param_dtype:
    x1 = x1.to(dtype=param_dtype)
    x3 = x3.to(dtype=param_dtype)
    t  = t.to(dtype=param_dtype)

# require gradients for spatial coords
x1.requires_grad_(True); x3.requires_grad_(True)

u1, u3, phi = net(x1, x3, t)
d = first_order_derivatives(u1, u3, phi, x1, x3)
c = constitutive(d)

# Evaluate tau0 at sample points
x3_np = x3.detach().cpu().numpy().ravel()
t_np = t.detach().cpu().numpy().ravel()
tau0_thermal = tau0_fn(x3_np, t_np).reshape(-1)

tau11 = c['tau11'].detach().cpu().numpy().ravel()

print('TAU_0_CONST =', cfg.TAU_0_CONST)
print('tau11: mean, min, max =', np.mean(tau11), np.min(tau11), np.max(tau11))
print('tau0_thermal: mean, min, max =', np.mean(tau0_thermal), np.min(tau0_thermal), np.max(tau0_thermal))
res = tau11 + cfg.TAU_0_CONST + tau0_thermal
print('residual: mean, min, max =', np.mean(res), np.min(res), np.max(res))
print('residual std=', np.std(res))

# Show a few samples
for i in range(10):
    print(i, 'tau11=', tau11[i], 'tau0=', tau0_thermal[i], 'res=', res[i])

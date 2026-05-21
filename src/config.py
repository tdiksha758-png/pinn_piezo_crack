"""Material constants (PZT-4) and simulation parameters.

All quantities in SI units unless noted.
References: Table 1 in the problem statement.
"""

from math import gamma as math_gamma

# ── Elastic stiffness (N m⁻²) ────────────────────────────────────────────────
MU_11: float = 139.0e9
MU_13: float = 74.3e9
MU_33: float = 113.0e9
MU_44: float = 25.6e9

# ── Piezoelectric constants (C m⁻²) ─────────────────────────────────────────
E_31: float = -6.98
E_33: float = 13.84
E_15: float = -13.44

# ── Dielectric permittivities (C V⁻¹ m⁻¹) ──────────────────────────────────
EPS_11: float = 60.0e-10
EPS_33: float = 54.7e-10

# ── Thermo-elastic coupling (N K⁻¹ m⁻²) ─────────────────────────────────────
KAPPA_11: float = 0.973e6
KAPPA_33: float = 0.791e6

# ── Pyroelectric coefficient (C K⁻¹ m⁻²) ───────────────────────────────────
P_Z: float = -48.86e-6

# ── Thermal / mechanical properties ─────────────────────────────────────────
RHO: float = 7.5e3          # density [kg m⁻³]
K_1: float = 0.52           # thermal conductivity, x₁-direction [W K⁻¹ m⁻¹]
K_3: float = 0.12           # thermal conductivity, x₃-direction [W K⁻¹ m⁻¹]
C_RHO: float = 420.0        # specific heat capacity [J kg⁻¹ K⁻¹]

# ── Initial / pre-stresses (N m⁻²) ──────────────────────────────────────────
SIGMA_11_0: float = 2.0e8
SIGMA_33_0: float = 2.0e8

# ── Geometry ─────────────────────────────────────────────────────────────────
H: float = 4.0              # strip height [m]
A_CRACK: float = 1.0        # lower crack-tip position  x₃ = a  [m]
B_CRACK: float = 3.0        # upper crack-tip position  x₃ = b  [m]
L_TRUNC: float = 10.0       # semi-infinite domain truncated at x₁ = L [m]

# ── Fractional heat model ─────────────────────────────────────────────────────
TAU_Q: float = 0.4          # thermal phase-lag [s]
GAMMA: float = 0.8          # Caputo order  0 < γ ≤ 1  (γ = 1 → classical CV)
T0_BC: float = 1.0          # Heaviside step amplitude  T(0,t) = T0 H(t)  [K]
T_MAX: float = 2.0          # simulation end time [s]

# ── Applied loading ──────────────────────────────────────────────────────────
TAU_0_CONST: float = 1.0e6  # uniform part of crack-face stress τ₀ [N m⁻²]

# ── Derived thermal diffusivity ──────────────────────────────────────────────
LAMBDA_0: float = K_3 / (RHO * C_RHO)   # effective diffusivity [m² s⁻¹]

# ── Effective moduli (Eqs. μ_E0, k_E0) ───────────────────────────────────────
_DENOM: float = MU_33 * EPS_33 + E_33 ** 2

MU_E0: float = MU_11 - (
    MU_13 * (MU_13 * EPS_33 + E_31 * E_33)
    + E_31 * (MU_13 * E_33 - MU_33 * E_31)
) / _DENOM

K_E0: float = KAPPA_11 - (
    KAPPA_33 * (MU_13 * EPS_33 + E_31 * E_33)
    - P_Z * (MU_13 * E_33 - MU_33 * E_31)
) / _DENOM

# Coefficient for fractional term in heat equation: C_γ = τq^γ / Γ(1+γ)
C_GAMMA: float = (TAU_Q ** GAMMA) / math_gamma(1.0 + GAMMA)

# ── PINN hyper-parameters ─────────────────────────────────────────────────────
# Collocation points
N_INTERIOR: int = 8_000     # interior (PDE) collocation points
N_BOUNDARY: int = 1_500     # points per boundary segment
N_IC: int = 500             # initial-condition points (t = 0)

# Network architecture  [input_dim, hidden, ..., output_dim]
# Mechanical PINN:  inputs (x1, x3, t) → outputs (u1, u3, φ)
MECH_LAYERS: list[int] = [3, 128, 128, 128, 128, 3]
# Temperature PINN: inputs (x3, t)     → output  T
TEMP_LAYERS: list[int] = [2, 64, 64, 64, 1]

# Training schedule
LR_ADAM: float = 1e-3
MAX_ITER_ADAM: int = 15_000
MAX_ITER_LBFGS: int = 5_000

# Loss weights
W_PDE: float = 1.0
W_BC: float = 10.0
W_IC: float = 10.0
W_FAR: float = 1.0          # far-field decay condition

# ── Normalisation scales (for numerical stability) ───────────────────────────
# Inputs scaled to [0, 1]:  x̄₁ = x₁/L_TRUNC, x̄₃ = x₃/H, t̄ = t/T_MAX
# Output reference scales
U_REF: float = H * TAU_0_CONST / MU_11       # displacement scale  [m]
PHI_REF: float = abs(E_15) * U_REF / EPS_11 / H   # electric potential  [V]
T_REF: float = T0_BC                          # temperature scale   [K]

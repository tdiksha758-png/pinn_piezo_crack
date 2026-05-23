# PINN Problem Flow

## Single-File Configuration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               src/problem_definition.py                         │
│                  ← ONLY THIS FILE CHANGES →                     │
│                                                                 │
│  DomainSpec         x1, x3, t bounds (t=None for steady)       │
│  BoundaryCondition  name + residual_fn + weight                 │
│  InitialCondition   name + residual_fn + weight  (optional)     │
│  ProblemSpec        PDE fn + BC list + IC list + params dict    │
│                                                                 │
│  ACTIVE_PROBLEM = ProblemSpec(...)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │  read by
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    src/loss.py    src/trainer.py   main.py
```

---

## End-to-End Flow

```
main.py
  │
  ├─ 1. Solve temperature (fractional CV heat equation)
  │       src/temperature.py  →  T(x₃, t)
  │
  ├─ 2. Compute thermal loading
  │       src/thermal_loading.py  →  τ₀(x₃, t)
  │       ACTIVE_PROBLEM.set_param("tau0_fn", tau0_fn)
  │             └── injects runtime value into problem spec
  │                 without editing problem_definition.py
  │
  ├─ 3. Construct network
  │       src/network.py  →  MechanicsNet
  │
  ├─ 4. Train
  │       src/trainer.py  →  Trainer(net)
  │             │
  │             └── calls pinn_loss() each step
  │                       │
  │                       └── src/loss.py  →  pinn_loss()
  │                                 │
  │                                 ├─ sample interior pts from domain
  │                                 ├─ ACTIVE_PROBLEM.pde_fn(...)  →  [R1,R2,R3]
  │                                 ├─ for bc in ACTIVE_PROBLEM.boundary_conditions:
  │                                 │       bc.residual_fn(net, N, device, dtype, **params)
  │                                 └─ for ic in ACTIVE_PROBLEM.initial_conditions:
  │                                         ic.residual_fn(...)   (skipped if None)
  │
  └─ 5. Post-process
          src/postprocess.py  →  plots, SIF
```

---

## Loss Composition

```
total_loss = w_pde × Σ‖PDE residuals‖²
           +        Σ (bc.weight × ‖BC residual‖²)
           + w_ic × Σ (ic.weight × ‖IC residual‖²)
                    └── term absent when initial_conditions = None
```

| Component | Controlled by |
|---|---|
| PDE weight (`w_pde`) | `Trainer(w_pde=...)` or `config.W_PDE` |
| Per-BC weight | `BoundaryCondition(..., weight=...)` in `problem_definition.py` |
| IC weight (`w_ic`) | `Trainer(w_ic=...)` or `config.W_IC` |
| Per-IC weight | `InitialCondition(..., weight=...)` in `problem_definition.py` |

---

## ProblemSpec Dataclass

```
ProblemSpec
├── name                  str
├── domain                DomainSpec
│     ├── x1_range        (float, float)
│     ├── x3_range        (float, float)
│     └── t_range         (float, float) | None   ← None = steady/quasi-static
│
├── pde_fn                Callable(net, x1, x3, t) → list[Tensor]
│
├── boundary_conditions   list[BoundaryCondition]
│     └── BoundaryCondition
│           ├── name          str
│           ├── residual_fn   Callable(net, N, device, dtype, **params) → Tensor | tuple
│           └── weight        float
│
├── initial_conditions    list[InitialCondition] | None
│     └── InitialCondition
│           ├── name          str
│           ├── residual_fn   Callable(net, N, device, dtype, **params) → Tensor
│           └── weight        float
│
└── params                dict   ← runtime values, set via set_param()
```

---

## Optional: Initial Conditions

| `initial_conditions` value | Behaviour |
|---|---|
| `None` | IC loss term = 0, no IC sampling |
| `[]` (empty list) | same as None |
| `[InitialCondition(...)]` | IC residuals computed and added to total loss |

---

## Residual Function Signatures

```python
# PDE
def my_pde_fn(net, x1: Tensor, x3: Tensor, t: Tensor) -> list[Tensor]:
    ...
    return [R1, R2, ...]          # one tensor per equation

# Boundary condition
def my_bc(net, N: int, device, dtype, **params) -> Tensor | tuple[Tensor, ...]:
    ...
    return residual               # or (r1, r2, r3) for multi-sub-conditions

# Initial condition  (only when problem is time-dependent)
def my_ic(net, N: int, device, dtype, **params) -> Tensor:
    ...
    return u_pred - u_exact       # residual at t = 0
```

---

## How to Switch Problems

**Edit only `src/problem_definition.py`**, then replace `ACTIVE_PROBLEM`:

```python
ACTIVE_PROBLEM = ProblemSpec(
    name="My New Problem",
    domain=DomainSpec(
        x1_range=(0.0, 1.0),
        x3_range=(0.0, 1.0),
        t_range=(0.0, 1.0),   # or None for steady
    ),
    pde_fn=my_pde_fn,
    boundary_conditions=[
        BoundaryCondition("bc1", my_bc, weight=10.0),
    ],
    initial_conditions=[          # or None
        InitialCondition("ic1", my_ic, weight=10.0),
    ],
)
```

**No changes needed in** `loss.py` · `trainer.py` · `main.py`.

---

## File Responsibilities

| File | Responsibility | Changes when problem changes? |
|---|---|---|
| `src/problem_definition.py` | Equations, BCs, ICs, domain | **Yes — only this one** |
| `src/config.py` | Material constants, hyperparameters | Only for material/training params |
| `src/pde_residuals.py` | PZT-4 PDE utility functions | No |
| `src/boundary_conditions.py` | PZT-4 BC utility functions | No |
| `src/loss.py` | Generic loss loop over problem spec | No |
| `src/trainer.py` | Adam + L-BFGS training loop | No |
| `src/network.py` | Neural network architectures | No |
| `main.py` | Orchestration, injects `tau0_fn` | No |

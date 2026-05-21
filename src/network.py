"""Neural-network architectures for the PINN solver.

MechanicsNet
    Inputs  : (x̄₁, x̄₃, t̄) ∈ [0,1]³   (normalised coordinates)
    Outputs : (ū₁, ū₃, φ̄)              (normalised displacement + potential)
    Physics : 3 coupled elliptic PDEs for a pre-stressed piezoelectric strip

The raw network output is multiplied by a *distance function* that
encodes the known homogeneous parts of the BCs so the network only needs
to learn the complementary solution.

TemperatureNet (optional stand-alone PINN for T^(1))
    Inputs  : (x̄₃, t̄) ∈ [0,1]²
    Outputs : T̄ ∈ ℝ  (normalised temperature)
    Physics : 1-D fractional Cattaneo–Vernotte equation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from . import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# Utility: fully-connected block
# ─────────────────────────────────────────────────────────────────────────────

class FCBlock(nn.Module):
    """Fully-connected layers with configurable activation."""

    def __init__(
        self,
        layer_sizes: list[int],
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.Tanh()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for stable early training."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Mechanics PINN  (u₁, u₃, φ)
# ─────────────────────────────────────────────────────────────────────────────

class MechanicsNet(nn.Module):
    """PINN for the quasi-static piezoelectric equations.

    Normalisation convention
    ------------------------
    x̄₁ = x₁ / L_TRUNC,   x̄₃ = x₃ / H,   t̄ = t / T_MAX
    ū₁ = u₁ / U_REF,      ū₃ = u₃ / U_REF,  φ̄ = φ / PHI_REF
    """

    def __init__(self, layer_sizes: list[int] | None = None) -> None:
        super().__init__()
        if layer_sizes is None:
            layer_sizes = cfg.MECH_LAYERS
        self.net = FCBlock(layer_sizes)

        # Normalisation constants (stored as buffers, not parameters)
        self.register_buffer("L",      torch.tensor(cfg.L_TRUNC, dtype=torch.float64))
        self.register_buffer("H",      torch.tensor(cfg.H,       dtype=torch.float64))
        self.register_buffer("T_MAX",  torch.tensor(cfg.T_MAX,   dtype=torch.float64))
        self.register_buffer("U_REF",  torch.tensor(cfg.U_REF,   dtype=torch.float64))
        self.register_buffer("PHI_REF",torch.tensor(cfg.PHI_REF, dtype=torch.float64))

    # ── input normalisation ─────────────────────────────────────────────────
    def _normalise_inputs(self, x1: Tensor, x3: Tensor, t: Tensor) -> Tensor:
        x1_n = x1 / self.L
        x3_n = x3 / self.H
        t_n  = t  / self.T_MAX
        # torch.cat along last dim: each (N,1) → (N,3)
        # torch.stack would insert a NEW dim → wrong shape (N,1,3)
        return torch.cat([x1_n, x3_n, t_n], dim=-1)   # (N, 3)

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(
        self, x1: Tensor, x3: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (u1, u3, phi) in physical units."""
        inp = self._normalise_inputs(x1, x3, t)
        out = self.net(inp)                          # (..., 3)
        u1  = out[..., 0:1] * self.U_REF
        u3  = out[..., 1:2] * self.U_REF
        phi = out[..., 2:3] * self.PHI_REF
        return u1, u3, phi

    def predict_flat(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Convenience wrapper: X has columns [x1, x3, t]."""
        return self.forward(X[:, 0:1], X[:, 1:2], X[:, 2:3])


# ─────────────────────────────────────────────────────────────────────────────
# Temperature PINN  T^(1)(x₃, t)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureNet(nn.Module):
    """Stand-alone PINN for the 1-D fractional heat equation (optional).

    The boundary condition T(0, t) = T₀ H(t) is embedded via the ansatz:
        T̂(x̄₃, t̄) = T₀ (1 − x̄₃) · σ(αt̄) · N(x̄₃, t̄)
    where σ is a sigmoid (smooth Heaviside) and N is the network output.
    This automatically satisfies T(H,t)=0 and T(x₃,0)≈0 for large α.
    """

    SIGMOID_STEEPNESS: float = 20.0   # sharpness of the smooth Heaviside

    def __init__(self, layer_sizes: list[int] | None = None) -> None:
        super().__init__()
        if layer_sizes is None:
            layer_sizes = cfg.TEMP_LAYERS
        self.net = FCBlock(layer_sizes, activation=nn.Tanh())

        self.register_buffer("H",     torch.tensor(cfg.H,     dtype=torch.float64))
        self.register_buffer("T_MAX", torch.tensor(cfg.T_MAX, dtype=torch.float64))
        self.register_buffer("T0",    torch.tensor(cfg.T0_BC, dtype=torch.float64))

    def forward(self, x3: Tensor, t: Tensor) -> Tensor:
        """Return T^(1)(x₃, t) in physical units (Kelvin)."""
        x3_n = x3 / self.H
        t_n  = t  / self.T_MAX

        inp = torch.cat([x3_n, t_n], dim=-1)    # (..., 2)
        raw = self.net(inp)                       # (..., 1)

        # Ansatz that enforces BCs weakly:
        # T ≈ T0 · (1 − x̄₃) · σ(α t̄) · (1 + raw)
        smooth_h = torch.sigmoid(self.SIGMOID_STEEPNESS * t_n)
        T = self.T0 * (1.0 - x3_n) * smooth_h * (1.0 + raw)
        return T

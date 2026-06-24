from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from . import config as cfg


# ============================================================
# FULLY CONNECTED BLOCK
# ============================================================

class FCBlock(nn.Module):

    def __init__(
        self,
        layer_sizes,
        activation=None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.Tanh()

        layers = []

        for i in range(len(layer_sizes) - 1):

            layers.append(
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1]
                )
            )

            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):

        for m in self.net.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        return self.net(x)


# ============================================================
# MECHANICS NETWORK
# ============================================================

class MechanicsNet(nn.Module):

    """
    Inputs:
        x1, x3, t

    Outputs:
        u1, u2, u3, phi
    """

    def __init__(self, layer_sizes=None):

        super().__init__()

        if layer_sizes is None:
            layer_sizes = cfg.MECH_LAYERS

        self.net = FCBlock(layer_sizes)

        self.register_buffer(
            "X1_MAX",
            torch.tensor(
                cfg.X1_MAX,
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "X3_MAX",
            torch.tensor(
                cfg.X3_MAX,
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "T_MAX",
            torch.tensor(
                cfg.T_MAX,
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "U_REF",
            torch.tensor(
                cfg.U_REF,
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "PHI_REF",
            torch.tensor(
                cfg.PHI_REF,
                dtype=torch.float64,
            ),
        )

    # ========================================================
    # INPUT NORMALIZATION
    # ========================================================

    def _normalise_inputs(
        self,
        x1: Tensor,
        x3: Tensor,
        t: Tensor,
    ) -> Tensor:

        x1_n = x1 / self.X1_MAX
        x3_n = x3 / self.X3_MAX
        t_n = t / self.T_MAX

        return torch.cat(
            [x1_n, x3_n, t_n],
            dim=-1,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x1: Tensor,
        x3: Tensor,
        t: Tensor,
    ):

        inp = self._normalise_inputs(
            x1,
            x3,
            t,
        )

        out = self.net(inp)

        u1 = out[:, 0:1] * self.U_REF

        u2 = out[:, 1:2] * self.U_REF

        u3 = out[:, 2:3] * self.U_REF

        phi = out[:, 3:4] * self.PHI_REF

        return (
            u1,
            u2,
            u3,
            phi,
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict_flat(
        self,
        X: Tensor,
    ):

        return self.forward(
            X[:, 0:1],
            X[:, 1:2],
            X[:, 2:3],
        )
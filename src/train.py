import torch
import matplotlib.pyplot as plt

from src import config as cfg

from src.network import MechanicsNet
from src.loss import pinn_loss


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dtype = torch.float64

print("Device :", device)


# ==========================================================
# NETWORKS
# ==========================================================

net_lower = MechanicsNet().to(
    device=device,
    dtype=dtype,
)

net_upper = MechanicsNet().to(
    device=device,
    dtype=dtype,
)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(
    list(net_lower.parameters())
    + list(net_upper.parameters()),
    lr=cfg.LR,
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.9995,
)


# ==========================================================
# TRAINING HISTORY
# ==========================================================

history = []


# ==========================================================
# TRAINING
# ==========================================================

print("\nStarting training...\n")

for step in range(1, cfg.N_ITER + 1):

    optimizer.zero_grad()

    loss = pinn_loss(
        net_lower,
        net_upper,
        N_int=cfg.N_INT,
        N_interface=cfg.N_INTERFACE,
        device=device,
        dtype=dtype,
    )

    loss.backward()

    optimizer.step()

    scheduler.step()

    history.append(loss.item())

    if step % 100 == 0 or step == 1:

        print(
            f"Step {step:6d}"
            f" | Loss = {loss.item():.6e}"
            f" | LR = {scheduler.get_last_lr()[0]:.3e}"
        )

print("\nTraining Finished.\n")


# ==========================================================
# SAVE MODELS
# ==========================================================

torch.save(
    net_lower.state_dict(),
    "net_lower.pt",
)

torch.save(
    net_upper.state_dict(),
    "net_upper.pt",
)

print("Models saved.")


# ==========================================================
# LOSS CURVE
# ==========================================================

plt.figure(figsize=(8,5))

plt.semilogy(history)

plt.xlabel("Iteration")

plt.ylabel("Loss")

plt.title("PINN Training Loss")

plt.grid(True)

plt.tight_layout()

plt.show()
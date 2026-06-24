import torch

def gradients(u, x):
    return torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )[0]

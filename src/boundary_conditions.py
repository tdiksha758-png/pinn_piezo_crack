
import torch

from . import config as cfg

from .constitutive import (
    first_order_derivatives,
    constitutive,
)

def _require_grad(*tensors):
    for t in tensors:
        t.requires_grad_(True)
def sample_interface(N, device, dtype):
    x = (
    cfg.X1_MIN
    + (cfg.X1_MAX - cfg.X1_MIN)
      * torch.rand(N,1,device=device,dtype=dtype)
)
    z = torch.zeros(N,1,device=device,dtype=dtype)
    t = torch.rand(N,1,device=device,dtype=dtype) * cfg.T_MAX

    return x, z, t
def bc_u1_continuity(net_lower, net_upper,
                     N, device, dtype):

    x, z, t = sample_interface(N, device, dtype)

    u1_l, u2_l, u3_l, phi_l = net_lower(x,z,t)
    u1_u, u2_u, u3_u, phi_u = net_upper(x,z,t)

    return u1_l - u1_u
def bc_u2_continuity(net_lower, net_upper,
                     N, device, dtype):

    x, z, t = sample_interface(N, device, dtype)

    u1_l, u2_l, u3_l, phi_l = net_lower(x,z,t)
    u1_u, u2_u, u3_u, phi_u = net_upper(x,z,t)

    return u2_l - u2_u
def bc_u3_continuity(net_lower, net_upper,
                     N, device, dtype):

    x, z, t = sample_interface(N, device, dtype)

    u1_l, u2_l, u3_l, phi_l = net_lower(x,z,t)
    u1_u, u2_u, u3_u, phi_u = net_upper(x,z,t)

    return u3_l - u3_u
def bc_T31_continuity(net_lower, net_upper,
                      N, device, dtype):

    x, z, t = sample_interface(N, device, dtype)

    _require_grad(x, z)

    u1_l, u2_l, u3_l, phi_l = net_lower(x,z,t)
    u1_u, u2_u, u3_u, phi_u = net_upper(x,z,t)

    d_l = first_order_derivatives(u1_l,u2_l,u3_l,phi_l,x,z)
    d_u = first_order_derivatives(u1_u,u2_u,u3_u,phi_u,x,z)

    c_l = constitutive(d_l)
    c_u = constitutive(d_u)

    return c_l["T31"] - c_u["T31"]
def bc_T32_continuity(net_lower, net_upper,
                      N, device, dtype):

    x, z, t = sample_interface(N, device, dtype)

    _require_grad(x, z)

    u1_l, u2_l, u3_l, phi_l = net_lower(x,z,t)
    u1_u, u2_u, u3_u, phi_u = net_upper(x,z,t)

    d_l = first_order_derivatives(u1_l,u2_l,u3_l,phi_l,x,z)
    d_u = first_order_derivatives(u1_u,u2_u,u3_u,phi_u,x,z)

    c_l = constitutive(d_l)
    c_u = constitutive(d_u)

    return c_l["T32"] - c_u["T32"]
def bc_T33_continuity(net_lower, net_upper,
                      N, device, dtype):

    x, z, t = sample_interface(N, device, dtype)

    _require_grad(x, z)

    u1_l, u2_l, u3_l, phi_l = net_lower(x,z,t)
    u1_u, u2_u, u3_u, phi_u = net_upper(x,z,t)

    d_l = first_order_derivatives(
        u1_l, u2_l, u3_l, phi_l, x, z
    )

    d_u = first_order_derivatives(
        u1_u, u2_u, u3_u, phi_u, x, z
    )

    c_l = constitutive(d_l)
    c_u = constitutive(d_u)

    return c_l["T33"] - c_u["T33"]
def bc_phi_continuity(net_lower, net_upper,
                      N, device, dtype):

    x, z, t = sample_interface(N, device, dtype)

    u1_l, u2_l, u3_l, phi_l = net_lower(x,z,t)
    u1_u, u2_u, u3_u, phi_u = net_upper(x,z,t)

    return phi_l - phi_u

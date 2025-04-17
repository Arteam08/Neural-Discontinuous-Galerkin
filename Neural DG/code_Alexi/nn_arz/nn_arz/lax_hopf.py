import torch
import numpy as np

def force_float64(func):
    def wrapper(*args, **kwargs):
        # Convert all tensor arguments to float64
        args = [arg.to(torch.float64) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(torch.float64) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

def bugers_q(k, flux):
    return -k**2/2

def burgers_qp(k, flux):
    return -k

def burgers_R(u, flux):
    return u**2/2

def burgers_Rp(u, flux):
    return u

burgers = {
    'v_max': 1.,#1/2,
    'w': -1.,#0,
    'q': bugers_q,
    'qp': burgers_qp,
    'R': burgers_R,
    'Rp': burgers_Rp
}

@force_float64
def Lax_Hopf_solver_convex(ic_ks, dx =.1, dt=.1, Nx=100, Nt=140, flow=burgers, device='cpu'):

  # Define initial conditions
    batch_size = ic_ks.shape[0]
    xmax = dx * Nx
    tmax = dt * Nt


    ic_xs = torch.linspace(0, xmax, ic_ks.shape[1]+1, device=device, dtype=torch.float64).unsqueeze(0).repeat(batch_size, 1)
    # ic_xs += torch.empty(batch_size, ic_xs.shape[1], device=device).uniform_(-(xmax/ic_ks.shape[1])/3, (xmax/ic_ks.shape[1])/3)
    # consider it as a problem on R
    ic_xs[:, 0] = -1e9
    ic_xs[:, -1] = 1e9

    ic_ks = ic_ks.to(device)

    # precompute bi's
    bi = []
    for i in range(ic_xs.shape[1] - 1):
        b = ic_ks[:, i] * ic_xs[:, i]
        for l in range(i):
            b -= (ic_xs[:, l + 1] - ic_xs[:, l]) * ic_ks[:, l]
        bi.append(b)
    bi = torch.stack(bi).to(device).T

    t = torch.arange(1e-9, tmax, dt, device=device, dtype=torch.float64).view(1, -1, 1, 1)
    x = torch.arange(0, xmax, dx, device=device, dtype=torch.float64).view(1, 1, -1, 1)

    xi = ic_xs.view(batch_size, 1, 1, -1)
    
    # conditionU = (xi > x + flow['w'] * t)
    conditionU = (xi > x - flow['v_max'] * t)
    # conditionJ = (xi < x - flow['v_max'] * t)
    conditionJ = (xi < x - flow['w'] * t)

    Jl = torch.where(
        conditionJ.any(dim=-1), 
        torch.clamp(
            # xi.shape[-1] - torch.argmax(conditionJ.float().flip(-1), dim=-1) - 1, 
            torch.argmax(conditionJ.float(), dim=-1) - 1, 
            max=xi.shape[-1] - 2,
            min=0
        ), 
        # torch.zeros_like(conditionU.any(dim=-1))
        len(xi)-2)*torch.ones_like(conditionJ.any(dim=-1)
    )
    
    Ju = torch.where(
        conditionU.any(dim=-1), 
        torch.clamp(
            xi.shape[-1] - torch.argmax(conditionU.int().flip(-1), dim=-1) - 1, 
            # torch.argmax(conditionU.int(), dim=-1) - 1, 
            max=xi.shape[-1] - 2,
            min=0
        ), 
        torch.zeros_like(conditionU.any(dim=-1))
        # len(xi)-2)*torch.ones_like(conditionU.any(dim=-1)
    )
    del conditionJ, conditionU

    #     Jl = torch.where(
    #     conditionJ.any(dim=-1), 
    #     torch.clamp(
    #         torch.argmax(conditionJ.float(), dim=-1) - 1, 
    #         min=0
    #     ), 
    #     len(xi)-2)*torch.ones_like(conditionJ.any(dim=-1)
    # )
    
    # Ju = torch.where(
    #     conditionU.any(dim=-1), 
    #     torch.clamp(
    #         xi.shape[-1] - torch.argmax(conditionU.int().flip(-1), dim=-1) - 1, 
    #         max=xi.shape[-1] - 2
    #     ), 
    #     torch.zeros_like(conditionU.any(dim=-1))
    # )

    xi = xi.expand(-1, t.shape[1], x.shape[2], -1)
    ki = ic_ks.view(batch_size, 1, 1, -1).expand(-1, t.shape[1], x.shape[2], -1)
    bi = bi.view(batch_size, 1, 1, -1).expand(-1, t.shape[1], x.shape[2], -1)

    max_range_len = (Ju - Jl + 1).max().item()
    i_range = Ju.unsqueeze(-1).expand(batch_size, t.shape[1], x.shape[2], max_range_len).clone()
    range_tensor = torch.arange(max_range_len, device=device).view(1, 1, 1, -1)
    i_range = torch.minimum(Jl.unsqueeze(-1) + range_tensor, Ju.unsqueeze(-1))
    del range_tensor, Ju, Jl

    xi_range_p1 = xi.gather(3, i_range + 1)
    xi_range = xi.gather(3, i_range)
    ki_range = ki.gather(3, i_range)
    bi_range = bi.gather(3, i_range)
    M_values = _Mc0(t, x, xi_range, xi_range_p1, ki_range, bi_range, flow)
    i_store = i_range.gather(3, torch.argmin(M_values, dim=-1, keepdim=True))
    del xi_range_p1, xi_range, ki_range, bi_range, M_values, i_range

    xip1 = xi.gather(3, i_store+1)
    xi = xi.gather(3, i_store)
    ki = ki.gather(3, i_store)
    del i_store
    rho = _rho_c0(t.squeeze(-1), x.squeeze(-1), xi.squeeze(-1), xip1.squeeze(-1), ki.squeeze(-1), flow=flow)
    return rho.cpu()

def _Mc0(t, x, xi, xip1, ki, bi, flow):
    c1 = xi + t * flow['w']
    c2 = xi + t * flow['qp'](ki, flow)
    c3 = xip1 + t * flow['qp'](ki, flow)
    c4 = xip1 + t * flow['v_max']

    M = torch.where(
        (x >= c1) & (x < c2),
        -t * flow['R']((x - xi) / t, flow) + ki * xi + bi,
        torch.where(
            (x >= c2) & (x < c3),
            +t * flow['q'](ki, flux=flow) - ki * x + bi,
            -t * flow['R']((x - xip1) / t, flow) + ki * xip1 + bi
            # torch.where(
            #     (x >= c3) & (x <= c4), 
            #     -t * flow['R']((x - xip1) / t, flow) + ki * xip1 + bi, 
            #     torch.full_like(M_condition_1, float('inf')))
        )
    )

    return M 

def _rho_c0(t, x, xi, xip1, ki, flow):
    t_flow_qp_ki = t * flow['qp'](ki, flux=flow)
    c1 = xi + t * flow['w']
    c2 = xi + t_flow_qp_ki
    c3 = xip1 + t_flow_qp_ki
    # c4 = xip1 + t * flow['v_max']

    return torch.where(
        (x >= c1) * (x < c2), 
        - flow['Rp']((x - xi)/t, flow),
        torch.where(
            (x >= c2) * (x < c3), 
            ki,
            - flow['Rp']((x - xip1)/t, flow)
            # - flow['Rp']((x - xip1)/t, flow)
            # torch.where(
            #     (x >= c3) * (x <= c4), 
            #     -flow['Rp']((x - xip1)/t, flow),
            #     torch.zeros_like(x)
            # )
        )
    )#.squeeze(-1)
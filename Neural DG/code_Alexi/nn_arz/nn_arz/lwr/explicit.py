import torch

def greenshieldRiemannSolution(x, t, c1, c2, v = 1, rmax = 4, f=None, fp=None, vf=None) -> float:
    # Greenshield_flux 
    if f == None:
        f = lambda r: v * r * (1- r/rmax)
        # derivative
        fp = lambda r: v * (1 - (2*r)/rmax)
        vf = lambda x_over_t :  0.5 * rmax * ( 1 - x_over_t.unsqueeze(0) / v)


    #  c1 < c2 : Simple shockwave
    slope = ((f(c2) - f(c1))/(c2-c1)).unsqueeze(1).unsqueeze(-1)

    result1 = torch.where(x.unsqueeze(-1) <= slope * t.unsqueeze(0), c1.unsqueeze(1).unsqueeze(-1)*torch.ones_like(x).unsqueeze(-1), c2.unsqueeze(1).unsqueeze(-1)*torch.ones_like(x).unsqueeze(-1))

    # c1 > c2 : Rarefaction wave
    s1 = fp(c1).unsqueeze(1).unsqueeze(-1)
    s2 = fp(c2).unsqueeze(1).unsqueeze(-1)

    val = vf(x.unsqueeze(-1) /(t.unsqueeze(0) + 1e-6))

    result2 = torch.where(x.unsqueeze(-1) <= s1*t.unsqueeze(0), c1.unsqueeze(1).unsqueeze(-1)*torch.ones_like(x).unsqueeze(-1), torch.where(x.unsqueeze(-1) > s2*t.unsqueeze(0), c2.unsqueeze(1).unsqueeze(-1)*torch.ones_like(x).unsqueeze(-1), val))

    result = torch.where((c1 <= c2).unsqueeze(-1).unsqueeze(-1), result1, result2)
    return result
import torch

def greenshieldFlux(rho, rho_max = 4., v_max = 1.):
    return (1 - rho/rho_max) * rho * v_max

def laxFriedrichs_flux_greenshield(rho_l, rho_r, rho_max = 4., v_max = 1.):
    fluxes_l = greenshieldFlux(rho_l)
    fluxes_r = greenshieldFlux(rho_r)
    return (fluxes_l + fluxes_r)/2 - torch.max(v_max*abs(1 - 2*fluxes_l/rho_max), v_max*abs(1 - fluxes_r/2)) * (fluxes_r - fluxes_l)/2

def godunov_flux_greenshield(rho_l, rho_r, rho_max = 4.):
    # print("WRONG")
    fluxes_l = greenshieldFlux(rho_l)
    fluxes_r = greenshieldFlux(rho_r)

    flows = torch.where(
        rho_l <= rho_r,
        torch.minimum(fluxes_l, fluxes_r),
        torch.where(
            rho_r > rho_max/2,
            # fluxes_r > rho_max/2,
            fluxes_r,
            torch.where(
                rho_max/2 > rho_l,  
                fluxes_l, 
                torch.where(
                    rho_l < rho_max/2,  
                    fluxes_l,
                    greenshieldFlux(torch.tensor(rho_max/2, device=rho_l.device)), 
                )
            )
        )
    )
    return flows

def flux(flux_function):
    if flux_function == 'laxfriedrichs':
        return laxFriedrichs_flux_greenshield
    elif flux_function == 'godunov':
        return godunov_flux_greenshield
    else:
        raise NotImplementedError
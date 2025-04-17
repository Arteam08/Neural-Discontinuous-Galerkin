import argparse
import torch
import os
import hashlib

import numpy as np 
import torch.nn.functional as F

import nn_arz.nn_arz.utils
import nn_arz.nn_arz.lwr.explicit
import nn_arz.nn_arz.lwr.fluxes
import nn_arz.nn_arz.lwr.lax_hopf
import nn_arz.nn_arz.polynomials

import time

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.6f} seconds")
        return result
    return wrapper

class Problem():
    def __init__(self, ic, solution, name):
        self.ic = ic # Liste ou tenseur selon les cas-pas utilisé pour l'instant
        self.solution = solution #Tensor(n_ic, n_points, n_time)
        self.name = name # string

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Eval script.")
    

    parser.add_argument("--cells", type=int, default=200, help="Number of cells.")
    parser.add_argument("--points_per_cell", type=int, default=42, help="Number of points per cell.")
    parser.add_argument("--number_of_polynomials", type=int, default=2, help="Number of polynomials.")
    parser.add_argument("--max_t", type=int, default=400, help="Maximum time.")
    parser.add_argument("--dx", type=float, default=1e-2, help="dx.")
    parser.add_argument("--dt", type=float, default=1e-3, help="dt.")

    parser.add_argument("--values_for_complex", type=int, default=10, help="Number of values for complex cases.")

    parser.add_argument("--seed", type=int, default=420, help="Seed for reproducibility.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU.")
    parser.add_argument("--device", type=str, default='None', help="Device.")

    parser.add_argument("--riemann", action="store_true", help="Solve Riemann problems.")
    parser.add_argument("--shock", action="store_true", help="Solve Shock problems.")
    parser.add_argument("--rarefaction", action="store_true", help="Solve Rarefaction problems.")
    parser.add_argument("--complex", action="store_true", help="Solve Complex problems.")
    parser.add_argument("--all", action="store_true", help="Solve all problems.")
    
    # Parse the arguments
    args = parser.parse_args()

    def hash_args(args):
        items = vars(args).items()
        args_str = ''
        for _, value in items:
            args_str += str(value)
        return hashlib.md5(args_str.encode()).hexdigest()

    save_folder = f'data/{hash_args(args)}'
    os.makedirs(save_folder, exist_ok=True)
    print(f'Saving values to {save_folder}')

    if args.all:
        args.riemann = True
        args.shock = True
        args.rarefaction = True
        args.complex = True
    
    # Set environment
    device = torch.device('cpu') if args.cpu else (args.device if args.device != "None" else nn_arz.nn_arz.utils.get_device())
    nn_arz.nn_arz.utils.seed(args.seed, deterministic=False)
    n_points = args.cells# * args.points_per_cell
    x_max = args.dx * n_points / 2
    x = torch.linspace(-x_max, x_max, n_points, device=device)
    time = torch.arange(1, args.max_t + 1, device=device).unsqueeze(0) * args.dt

    print(f"Problem settings: dx={args.dx:.2e}, dt={args.dt:.2e}, max_t={args.max_t:.2e}, x_max={x_max:.2e}, n_points={n_points}")


    # Define initial conditions
    N = 15
    ic = np.linspace(0, 4., N)
    points = np.array([[c1, c2] for c1 in ic for c2 in ic if c1 != c2])
    points_shock = np.array([[c1, c2] for c1 in ic for c2 in ic if c1 < c2])
    points_rarefaction = np.array([[c1, c2] for c1 in ic for c2 in ic if c1 > c2])
    c1 = torch.tensor(points[:, 0], device=device).float()
    c2 = torch.tensor(points[:, 1], device=device).float()
    c1_shock = torch.tensor(points_shock[:, 0], device=device).float()
    c2_shock = torch.tensor(points_shock[:, 1], device=device).float()
    c1_rarefaction = torch.tensor(points_rarefaction[:, 0], device=device).float()
    c2_rarefaction = torch.tensor(points_rarefaction[:, 1], device=device).float()
    
    riemann_ic = [c1, c2]
    shock_ic = [c1_shock, c2_shock]
    rarefaction_ic = [c1_rarefaction, c2_rarefaction]

    complex_ic = 4*torch.rand((N*(N-1), args.values_for_complex), device=device)

    # solve Riemann problems
    if args.riemann:
        if os.path.exists(f'{save_folder}/riemann.pth'):
            solutions_riemann = torch.load(f'{save_folder}/riemann.pth', weights_only=True)
        else:
            solutions_riemann = nn_arz.nn_arz.lwr.explicit.greenshieldRiemannSolution(x, time, c1, c2)
            torch.save(solutions_riemann, f'{save_folder}/riemann.pth')
    if args.shock:
        if os.path.exists(f'{save_folder}/shock.pth'):
            solutions_shock = torch.load(f'{save_folder}/shock.pth', weights_only=True)
        else:
            solutions_shock = nn_arz.nn_arz.lwr.explicit.greenshieldRiemannSolution(
                x, 
                time,
                c1_shock, 
                c2_shock
            )
            torch.save(solutions_shock, f'{save_folder}/shock.pth')

    if args.rarefaction:
        if os.path.exists(f'{save_folder}/rarefaction.pth'):
            solutions_rarefaction = torch.load(f'{save_folder}/rarefaction.pth', weights_only=True)
        else:
            solutions_rarefaction = nn_arz.nn_arz.lwr.explicit.greenshieldRiemannSolution(
                x, 
                time,
                c1_rarefaction, 
                c2_rarefaction
            )
            torch.save(solutions_rarefaction, f'{save_folder}/rarefaction.pth')

    # Solve complex initial conditions
    if args.complex:
        if os.path.exists(f'{save_folder}/complex.pth'):
            solution_complex = torch.load(f'{save_folder}/complex.pth', weights_only=True)
        else:
            ratio=10 # nombre de points pour calculer la moyenne dans l'intervalle avel lax_hopf=quadrature
            batch_size_complex = 1
            solution_complex = torch.empty((complex_ic.shape[0], n_points, args.max_t))
            for i in range(complex_ic.shape[0]//batch_size_complex):
                temp = nn_arz.nn_arz.lwr.lax_hopf.Lax_Hopf_solver_Greenshield(
                    complex_ic[i*batch_size_complex:(i+1)*batch_size_complex], 
                    dx=args.dx/ratio, 
                    dt=args.dt, 
                    Nx=n_points*ratio, 
                    Nt=args.max_t, 
                    device=device
                )
                solution_complex[i*batch_size_complex:(i+1)*batch_size_complex] = temp.reshape((
                    temp.shape[0], 
                    temp.shape[1], 
                    temp.shape[2] // ratio, 
                    ratio)
                ).mean(dim=3).transpose(1, 2)

            torch.save(solution_complex, f'{save_folder}/complex.pth')
    

    problems = []
    if args.riemann:
        problems.append(Problem(riemann_ic, solutions_riemann.cpu(), "Riemann"))
    if args.shock:
        problems.append(Problem(shock_ic, solutions_shock.cpu(), "Shock"))
    if args.rarefaction:
        problems.append(Problem(rarefaction_ic, solutions_rarefaction.cpu(), "Rarefaction"))
    if args.complex:
        problems.append(Problem(complex_ic, solution_complex.cpu(), "Complex"))


    ##### Variables for Discontinuous Galerkin
    half_cell_size = x_max/args.cells
    polynomials = nn_arz.nn_arz.polynomials.basis(
        'legendre', 
        args.number_of_polynomials, 
        device=device
    )
    # Pre-compute constants
    mass_matrix = polynomials.mass_matrix() * half_cell_size # * h because of the rescale from [-1, 1] to [-x_max, x_max]/cells
    mass_matrix_inv = (mass_matrix**(-1)).unsqueeze(0).unsqueeze(-1)

    x_polynomials, weights_leggauss = nn_arz.nn_arz.utils.legendre_roots_weights(
        args.points_per_cell, 
        device=device
    )
    weights_leggauss = weights_leggauss.unsqueeze(0).unsqueeze(0)
    
    polynomials_prime = polynomials.prime()(x_polynomials)
    polynomials_prime = polynomials_prime.unsqueeze(0) / half_cell_size # rescale the derivative to the new domain
    polynomials = polynomials(x_polynomials).unsqueeze(0)
    
    left_polynomials_value = torch.tensor([1, -1], device=device).repeat(args.number_of_polynomials // 2 + 1)[:args.number_of_polynomials].unsqueeze(0).unsqueeze(-1)

    time = torch.arange(1, args.max_t + 1, device=device).unsqueeze(0) * args.dt

    ######## Use numerical methods
    with torch.no_grad():
        for problem in problems:
            problem.solution = problem.solution.to(device)

            if os.path.exists(f'{save_folder}/{problem.name}_godunov.pth'):
                solution_godunov = torch.load(
                    f'{save_folder}/{problem.name}_godunov.pth', 
                    weights_only=True,
                    map_location=device
                    )
                compute_godunov = False
            else:
                solution_godunov = problem.solution.clone()
                compute_godunov = True


            if os.path.exists(f'{save_folder}/{problem.name}_DG.pth'):
                solution_DG = torch.load(
                    f'{save_folder}/{problem.name}_DG.pth', 
                    weights_only=True, 
                    map_location=device
                )
                compute_DG = False
            else:
                solution_DG = torch.nn.functional.interpolate(
                    problem.solution.clone().unsqueeze(1), 
                    size=(args.points_per_cell*args.cells, args.max_t), 
                    mode='bilinear'
                ).squeeze(1)
                compute_DG = True


            batch_size = problem.solution.shape[0]
            weights_dg = torch.empty(
                batch_size, 
                args.number_of_polynomials, 
                args.cells, 
                device=device)
            
            if compute_DG:
                ##### Initial weights Discontinuous Galerkin
                for cell in range(args.cells):
                    indices = slice(cell * args.points_per_cell, (cell + 1) * args.points_per_cell)
                    
                    # weights_dg[:, :, cell] = mass_matrix_inv.squeeze(-1) * torch.einsum('ijk,ik->ij', polynomials, solution_DG[:, indices, 0]) * args.dx
                    if compute_DG:

                        #_______initilalise weights________
                        weights_dg[:, :, cell] = mass_matrix_inv.squeeze(-1) * (weights_leggauss * (polynomials * solution_DG[:, indices, 0].unsqueeze(1))).sum(dim=-1) * half_cell_size
                        
                if compute_DG:
                    solution_DG[:, :, 0] = torch.einsum(
                        'ijk,ijl->ikl', 
                        polynomials, 
                        weights_dg
                    ).permute(0, 2, 1).reshape(batch_size, -1)

                left_boundary_indexes = torch.arange(
                    -1, 
                    args.cells*args.points_per_cell, 
                    args.points_per_cell
                ) 
                right_boundary_indexes = torch.arange(
                    0, 
                    args.cells*args.points_per_cell+1, 
                    args.points_per_cell
                )
                left_boundary_indexes[0] = 0
                right_boundary_indexes[-1] = args.cells*args.points_per_cell- 1
            
            if compute_godunov:
                for t in range(1, args.max_t):
                    ##### Godunov
                    fluxes_riemann = nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield(solution_godunov[:, :-1, t-1], solution_godunov[:, 1:, t-1])
                    fluxes_riemann = torch.cat([fluxes_riemann[:, [0]], fluxes_riemann, fluxes_riemann[:, [-1]]], dim=1)
                            
                    solution_godunov[:, 1:-1, t] = solution_godunov[:, 1:-1, t-1] - (fluxes_riemann[:, 2:-1] - fluxes_riemann[:, 1:-2]) * args.dt / args.dx 
                    # Boundaries: fixed to solution
                    #####

            if compute_DG:
                ##### Discontinuous Galerkin
                DG_update(
                    args.max_t, 
                    solution_DG, 
                    problem.solution,
                    weights_dg, 
                    left_boundary_indexes,
                    right_boundary_indexes, 
                    nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield, 
                    mass_matrix_inv,
                    half_cell_size,
                    batch_size,
                    args.cells,
                    args.points_per_cell,
                    weights_leggauss,
                    polynomials,
                    polynomials_prime,
                    left_polynomials_value,
                    args.dt
                )

            if compute_DG:
                solution_DG = (solution_DG.transpose(1, 2) * weights_leggauss.repeat(1, 1, args.cells)).view((
                        solution_DG.shape[0], 
                        solution_DG.shape[2], 
                        solution_DG.shape[1] // args.points_per_cell, 
                        args.points_per_cell)
                    ).sum(dim=3).transpose(1, 2) / 2 #intégrale sur chaque cell en post processing (instable sans ça)

            n = solution_godunov.shape[0] 
            mse_godunov = F.mse_loss(solution_godunov, problem.solution, reduction='none').mean(dim=1).mean(dim=1)
            mse_DG = F.mse_loss(solution_DG, problem.solution, reduction='none').mean(dim=1).mean(dim=1)

            print(f"Problem: {problem.name}")
            print(f"\tGodunov:      {mse_godunov.mean():.3e} ± {mse_godunov.std():.1e},      {100*(mse_godunov <= mse_DG).sum()//n:3d}%")
            print(f"\tDG:           {mse_DG.mean():.3e} ± {mse_DG.std():.1e}, {100*(mse_DG <= mse_godunov).sum()//n:3d}%")     

            problem.solution = problem.solution.cpu()
            to_display = torch.stack([
                solution_godunov.cpu()[[1,5,81]], 
                solution_DG.cpu()[[1,5,81]], 
                problem.solution[[1, 5, 81]],
                abs(solution_godunov.cpu()[[1,5,81]] - problem.solution[[1, 5, 81]]),
                abs(solution_DG.cpu()[[1,5,81]] - problem.solution[[1, 5, 81]])]
            ).transpose(0, 1).squeeze(-1)
            nn_arz.nn_arz.utils.plot_matrices(
                to_display, 
                titles=[
                    'Godunov', 
                    'DG', 
                    'Ground Truth',
                    'Abs diff Godunov',
                    'Abs diff DG'
                ], 
                xlabel='rho', 
                ylabel='t', 
                cmap='jet', 
                extent=[-x_max, x_max, 0, args.max_t*args.dt], 
                file=f'output{problem.name}.pdf'
            )

            import matplotlib.pyplot as plt
            plt.plot(problem.solution[5, :, 0].cpu(), label='IC')
            plt.plot(solution_DG[5, :, -1].cpu(), label='DG')
            plt.plot(solution_godunov[5, :, -1].cpu(), label='Godunov')
            plt.legend()
            plt.savefig(f'output{problem.name}_profiles.pdf')
            plt.clf()

            if compute_godunov:
                torch.save(solution_godunov, f'{save_folder}/{problem.name}_godunov.pth')
            if compute_DG:
                torch.save(solution_DG, f'{save_folder}/{problem.name}_DG.pth')

            del solution_godunov, solution_DG
            del problem.solution

def DG_update(
        max_t, 
        solution_DG, #shape : (n_ic, N_cells*points_per_cell, n_time), currently an interpoleted version of the ground truth
        solution,
        weights_dg, # shape : (n_ic, n_polynomials, N_cells) correspond aux weights à la première timestep
        left_boundary_indexes,   # u_{i-1}+ point extrême répété (ghost cell)
        right_boundary_indexes,  # u_{i}-
        flow_func, #takes the left and right values at the boundary
        mass_matrix_inv,
        half_cell_size,
        batch_size,
        cells, #number of cells
        points_per_cell,
        weights_leggauss,
        polynomials,
        polynomials_prime,
        left_polynomials_value,
        dt):
    """Modifies the solution_DG in place"""
    print("weights_dg",weights_dg.shape)
    print("solution DG",solution_DG.shape)
    print("t=0",solution_DG[0,:,0])
    print("t=1",solution_DG[0,:,1])
    
    for t in range(1, max_t):
        weights_dg_save = weights_dg.clone() #Ut
        for a in range(2):
            #RK2
            #U{t-1} si a =0, U{t} si a=1
            left_boundaries = solution_DG[:, left_boundary_indexes, t+a-1]#
            right_boundaries = solution_DG[:, right_boundary_indexes, t+a-1]

            fluxes = flow_func(left_boundaries, right_boundaries) # à modifier pour accpeter des formes plus générales
            # For legendre, right boundary is always 1 and left is (-1)^n
            fluxes = fluxes[:, 1:].unsqueeze(1) - fluxes[:, :-1].unsqueeze(1) * left_polynomials_value
                            
            fluxes = (mass_matrix_inv * fluxes)
                            
            f_u = nn_arz.nn_arz.lwr.fluxes.greenshieldFlux(solution_DG[:, :, t+a-1])



            residual = (
                mass_matrix_inv * (
                    weights_leggauss.unsqueeze(-1) * (
                        polynomials_prime.unsqueeze(-1) * f_u.view(
                            batch_size, 
                            cells,
                            points_per_cell
                        ).transpose(1, 2).unsqueeze(1)
                    ) 
                ).sum(dim=2) * half_cell_size
            ).float() # M^-1*(f|phi')
            L=-fluxes+residual

                            
            weights_dg = weights_dg_save + (dt*L).squeeze(1) * (1/2 if a==0 else 1.) #weights_dg is not saved but used in the solution
            # code pour enforcer direct la solution aux bords
            # weights_dg[:, 1:, 0] = 0
            # weights_dg[:, 0, 0] = solution[:, 0, t+a-1]
            # weights_dg[:, 1:, -1] = 0
            # weights_dg[:, 0, -1] = solution[:, -1, t+a-1]

            solution_DG[:, :, t] = torch.einsum('ijk,ijl->ikl', polynomials, weights_dg).permute(0, 2, 1).reshape(batch_size, -1)

            
if __name__ == "__main__":
    main()
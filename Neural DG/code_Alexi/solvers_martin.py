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
import matplotlib.pyplot as plt
from generate_data_martin import *



###define default basis

import torch

def piecewise_constant_interpolate(x, target_size):
    """
    x: tensor of shape (n_ic, 1, num_pieces)
       where each value in the last dimension represents a constant segment
    target_size: int, desired output size along the last dimension
    
    returns: tensor of shape (n_ic, 1, target_size)
    """
    n_ic, _, num_pieces = x.shape

    # Determine segment lengths: how many times each piece should be repeated
    base_len = target_size // num_pieces
    remainder = target_size % num_pieces

    # Create the list of repeat counts (distribute the remainder)
    repeat_counts = [base_len + (1 if i < remainder else 0) for i in range(num_pieces)]

    # Repeat each value according to its repeat count
    segments = [
        x[:, :, i].unsqueeze(-1).expand(-1, -1, repeat_counts[i])
        for i in range(num_pieces)
    ]

    return torch.cat(segments, dim=-1)


def get_default_basis(n_poly, dx, points_per_cell, device):
    """gets all the feateures from the legengdre basis"""
    polynomials = nn_arz.nn_arz.polynomials.basis(
        'legendre', 
        n_poly, 
        device=device
    )
    half_cell_size = dx / 2
    # Pre-compute constants
    mass_matrix = polynomials.mass_matrix() * half_cell_size # * h because of the rescale from [-1, 1] to [-x_max, x_max]/cells
    mass_matrix_inv = (mass_matrix**(-1)).unsqueeze(0).unsqueeze(-1)

    x_polynomials, weights_leggauss = nn_arz.nn_arz.utils.legendre_roots_weights(
        points_per_cell, 
        device=device
    )
    weights_leggauss = weights_leggauss.unsqueeze(0).unsqueeze(0)

    polynomials_prime = polynomials.prime()(x_polynomials)
    polynomials_prime = polynomials_prime.unsqueeze(0) / half_cell_size # rescale the derivative to the new domain
    polynomials = polynomials(x_polynomials).unsqueeze(0)

    left_polynomials_value = torch.tensor([1, -1], device=device).repeat(n_poly// 2 + 1)[:n_poly].unsqueeze(0).unsqueeze(-1)
    right_polynomials_value = torch.ones_like(left_polynomials_value)
    return polynomials, polynomials_prime, mass_matrix, mass_matrix_inv, weights_leggauss, left_polynomials_value, right_polynomials_value

def soft_clamp_sp(x, a, b):
    # F.softplus(z) = log(1 + exp(z))
    return a + F.softplus(x - a) - F.softplus(x - b)





### define the solver class
class DG_solver():

    def __init__(self,t_max ,# time steps # time step
              dx, # space step
              dt,
              n_cells, # number of cells
              flow_func, 
              device,# flow function, exact arguments to be clarified
              points_per_cell=40, # number of points per cell, can be replaced by a quadrature for less space storage
              n_poly=2,
                ):

        self.t_max = t_max
        self.dx = dx # space step
        self.dt = dt # time step
        self.n_cells = n_cells
        self.flow_func = flow_func # functin we use to approximate the flux
        self.f_PDE= nn_arz.nn_arz.lwr.fluxes.greenshieldFlux # function inherent to the PDE
        self.points_per_cell = points_per_cell # points used in each cell to compute the average value over the cell in post processing
        self.n_poly=n_poly # basis size
        self.device = device
        left_boundary_indexes = torch.arange(
                    -1, 
                    self.n_cells*self.points_per_cell, 
                    self.points_per_cell
                ) 
        right_boundary_indexes = torch.arange(
                    0, 
                    self.n_cells*self.points_per_cell+1, 
                    self.points_per_cell
                )
        left_boundary_indexes[0] = 0
        right_boundary_indexes[-1] = n_cells*points_per_cell- 1
        self.left_boundary_indexes = left_boundary_indexes # indices pour trouver dans le tenseur de solution u_{i-1}+ point extrême répété (ghost cell)
        self.right_boundary_indexes = right_boundary_indexes # u_{i}-

        # Define the polynomial basis// can be generalized
        polynomials, polynomials_prime, mass_matrix, mass_matrix_inv, weights_leggauss, left_polynomials_value, right_polynomials_value = get_default_basis(self.n_poly, self.dx, self.points_per_cell ,self.device)
        self.quadrature_weights = weights_leggauss
        self.basis_func = polynomials #shape (n_poly, points_per_cell)
        self.basis_func_prime = polynomials_prime
        self.mass_matrix = mass_matrix
        self.mass_matrix_inv = mass_matrix_inv
        self.left_polynomials_value = left_polynomials_value
        self.right_polynomials_value = right_polynomials_value

    def process_initial_conditions(self, initial_conditions):
        
        """initialize the decompositons weights and solution tensor
        initial_conditions: tensor of shape (n_ic, some_discretization)
        """
        n_ic = initial_conditions.shape[0]

        #interpolte the inital conditon
        initial_conditions= initial_conditions.unsqueeze(1) #shape (n_ic, 1, some_discretization)
    
        # interp_ic=F.interpolate(initial_conditions, size=self.n_cells*self.points_per_cell, mode='linear', align_corners=True) # shape (n_ic, 1, n_cells*points_per_cell)

        #only for Riemann
        interp_ic = piecewise_constant_interpolate(initial_conditions, self.n_cells*self.points_per_cell) 

        interp_ic=interp_ic.squeeze(1) # shape (n_ic, n_cells*points_per_cell)
        #decompose it by cell
        interp_ic = interp_ic.view(n_ic, self.n_cells, self.points_per_cell)
        #initialize the solution tensor
        solution_DG = torch.zeros((n_ic, self.n_cells * self.points_per_cell, self.t_max), device=self.device) ### (n_ic, n_cells*points_per_cell, t_max), stores each value, in space and time
        
        #compute the weights on the initil condition --to be optimized
        weights_dg = torch.empty(
                n_ic, 
                self.n_poly, 
                self.n_cells, 
                device=self.device)


        temp = self.mass_matrix_inv * self.quadrature_weights * self.basis_func  
        # temp has shape [1, n_poly, points_per_cell]; squeeze the singleton batch dimension:
        temp = temp.squeeze(0)  # now shape [n_poly, points_per_cell]

        weights_dg = torch.einsum('lp,icp->ilc', temp, interp_ic) * (self.dx / 2)

        reconstructed_solution = torch.einsum('ilc,lk->ick', weights_dg, self.basis_func.squeeze(0))
        reconstructed_solution = reconstructed_solution.reshape(n_ic, self.n_cells * self.points_per_cell)
        solution_DG[:, :, 0] = reconstructed_solution

        return weights_dg, solution_DG, n_ic

    
    def compute_L(self,solution_DG, t, a):
        n_ic = solution_DG.shape[0]

        u_prev = solution_DG[:, :, t+a-1]
        # this is done to avoid the in-place operation and let the gradients be computed
        u_prev = u_prev.clone()


        ##############
        ###Artificial stabilization####@@@@
        u_max=4
        u_prev=u_prev.clamp(min=-1, max=5) # to avoid the NaN, it is observed that enabling small excesses improves performance // to be made deiiferntiable



        assert not torch.isnan(u_prev).any(), "u_prev is NaN"
        # print("u_prev",torch.max(u_prev).item(), torch.min(u_prev).item())

        left_boundaries  = u_prev[:, self.left_boundary_indexes]
        right_boundaries = u_prev[:, self.right_boundary_indexes]
        f_u   = self.f_PDE(u_prev)
        assert not torch.isnan(f_u).any(), "f_u is NaN"
        # print("f_u",torch.max(f_u).item(), torch.min(f_u).item())
  

        fluxes = self.flow_func(left_boundaries, right_boundaries) # à modifier pour accpeter des formes plus générales
        assert not torch.isnan(fluxes).any(), "fluxes is NaN"
        # print("fluxes",torch.max(fluxes).item(), torch.min(fluxes).item())

        # print("uvalues",torch.max(left_boundaries), torch.min(left_boundaries))

        # !! Right polynomial values not implemented
        fluxes = fluxes[:, 1:].unsqueeze(1)*self.right_polynomials_value - fluxes[:, :-1].unsqueeze(1) * self.left_polynomials_value
                        
        fluxes = (self.mass_matrix_inv * fluxes)

        # plt.plot(fluxes[0, 1,:].cpu().detach().numpy())
        # plt.show()
                        
            
        residual = (
            self.mass_matrix_inv * (
                self.quadrature_weights.unsqueeze(-1) * (
                    self.basis_func_prime.unsqueeze(-1) * f_u.view(
                        n_ic, 
                        self.n_cells,
                        self.points_per_cell
                    ).transpose(1, 2).unsqueeze(1)
                ) 
            ).sum(dim=2) * self.dx / 2
        ).float() # M^-1*(f|phi')
        assert not torch.isnan(residual).any(), "residual is NaN"
        L=-fluxes+residual
        return L
    
    def solve(self, initial_conditions):
        weights_dg, solution_DG, n_ic = self.process_initial_conditions(initial_conditions)
        for t in range(1, self.t_max):
            weights_dg_save = weights_dg.clone() #Ut
            for a in range(2):
                #RK2
                #U{t-1} si a =0, U{t} si a=1
                L=self.compute_L(solution_DG, t, a)
                assert not torch.isnan(L).any(), "L is NaN"

        
                weights_dg = weights_dg_save + (self.dt*L).squeeze(1) * (1/2 if a==0 else 1.) #weights_dg is not saved but used in the solution

                solution_DG[:, :, t] = torch.einsum('ijk,ijl->ikl', self.basis_func, weights_dg).permute(0, 2, 1).reshape(n_ic, -1)
                assert not torch.isnan(solution_DG).any(), "RK is NaN"
        return solution_DG


    
    def cell_averaging(self, solution_dg):
        """post processing of the solution_DG tensor to aveage it over each cell"""
         #solution_dg is of shape (n_cells, n_cells*points_per_cell, t_max), we want to average over the points_per_cell
        ic= solution_dg.shape[0]
        solution_dg = solution_dg.reshape(ic, self.n_cells, self.points_per_cell, self.t_max)
        solution_dg = solution_dg.mean(dim=2)
        return solution_dg
    
    ####Running and plotting######
    def plot_solver(self, dataset_path, batch_size, plot_path):
        """
        Run the solver, compute per-sample absolute & relative L2 errors in one pass,
        and output comparison heatmaps for the first 3 ICs.
        """
        device = self.device

        # Load dataset
        ds = HyperbolicDataset(dataset_path)
        N = len(ds)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # Pre-allocate error tensors
        l2_errors      = torch.empty(N, device=device)  # (N,)
        rel_l2_errors  = torch.empty(N, device=device)  # (N,)

        l1_errors      = torch.empty(N, device=device)
        rel_l1_errors  = torch.empty(N, device=device)

        # Batch-wise solve & error
        for batch_idx, batch in enumerate(loader):
            start = batch_idx * batch_size
            end   = start + batch['ic'].shape[0]

            ic         = batch['ic'].to(device)         # (B, 2)
            sol_exact  = batch['sol_exact'].to(device)  # (B, C, T)

            # DG solve + averaging → (B, C, T)
            sol_DG_large = self.solve(ic)
            sol_DG       = self.cell_averaging(sol_DG_large)

            # flatten spatial/time dims
            diff = sol_DG - sol_exact                  # (B, C, T)
            flat_diff = diff.view(diff.shape[0], -1)   # (B, C*T)
            flat_exact = sol_exact.view(sol_exact.shape[0], -1)  # (B, C*T)

            # absolute L2 per sample
            l2_batch = torch.sqrt((flat_diff**2).sum(dim=1))       # (B,)
            # exact-solution L2 per sample
            norm_exact = torch.sqrt((flat_exact**2).sum(dim=1))   # (B,)
            # relative L2 = abs / exact
            rel_l2_batch = l2_batch / (norm_exact +1e-8)                 # (B,)
            l1_batch = flat_diff.abs().sum(dim=1)                  # (B,)
            l1_exact = flat_exact.abs().sum(dim=1)                 # (B,)
            rel_l1_batch = l1_batch / (l1_exact + 1e-8)            # avoid divide-by-zero


            # store
            l2_errors[start:end]     = l2_batch
            rel_l2_errors[start:end] = rel_l2_batch

            l1_errors[start:end]     = l1_batch
            rel_l1_errors[start:end] = rel_l1_batch


        # --- Plot heatmaps for the first 3 ICs ---
        full_data = torch.load(dataset_path)
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        for i in range(3):
            exact = full_data['sol_exact'][i].cpu().numpy()  # (C, T)
            dg    = self.cell_averaging(
                        self.solve(full_data['ic'][i:i+1].to(device))
                    )[0].cpu().numpy()  # (C, T)

            ax_e, ax_d = axes[i, 0], axes[i, 1]
            ax_e.imshow(exact, aspect='auto')
            ax_e.set(title=f'Exact IC #{i}', xlabel='Time', ylabel='Cell')
            ax_d.imshow(dg, aspect='auto')
            ax_d.set(title=f'DG IC #{i}', xlabel='Time', ylabel='Cell')

        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)

        # report mean errors
        mean_abs_l2 = l2_errors.mean().item()
        mean_rel_l2 = rel_l2_errors.mean().item()
        mean_abs_l1 = l1_errors.mean().item()
        mean_rel_l1 = rel_l1_errors.mean().item()

        print(f"Saved comparison heatmaps to {plot_path}")
        print(f"Mean absolute L2 error: {mean_abs_l2:.4e}")
        print(f"Mean relative L2 error: {mean_rel_l2:.4e}")
        print(f"Mean absolute L1 error: {mean_abs_l1:.4e}")
        print(f"Mean relative L1 error: {mean_rel_l1:.4e}")


    def run_solver(self, dataset_path, batch_size):
        """outputs the solution tensor"""
        ds = HyperbolicDataset(dataset_path)
        N = len(ds)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for batch_idx, batch in enumerate(loader):
            start = batch_idx * batch_size
            end   = start + batch['ic'].shape[0]

            ic         = batch['ic'].to(self.device)         # (B, 2)
            sol_exact  = batch['sol_exact'].to(self.device)  # (B, C, T)

            # DG solve + averaging → (B, C, T)
            sol_DG_large = self.solve(ic)
            sol_DG       = self.cell_averaging(sol_DG_large)
            return sol_DG


    def compute_metrics(self, loader):
        """commutes L1 and L2 performance from a dataloader"""
        N=len(loader.dataset)
        l2_errors      = torch.empty(N, device=self.device)  # (N,)
        rel_l2_errors  = torch.empty(N, device=self.device)  # (N,)

        l1_errors      = torch.empty(N, device=self.device)
        rel_l1_errors  = torch.empty(N, device=self.device)

        # Batch-wise solve & error
        for batch_idx, batch in enumerate(loader):
            start = batch_idx * loader.batch_size
            end   = start + batch['ic'].shape[0]

            ic         = batch['ic'].to(self.device)         # (B, 2)
            sol_exact  = batch['sol_exact'].to(self.device)  # (B, C, T)

            # DG solve + averaging → (B, C, T)
            sol_DG_large = self.solve(ic)
            sol_DG       = self.cell_averaging(sol_DG_large)

            # flatten spatial/time dims
            diff = sol_DG - sol_exact                  # (B, C, T)
            flat_diff = diff.view(diff.shape[0], -1)   # (B, C*T)
            flat_exact = sol_exact.view(sol_exact.shape[0], -1)  # (B, C*T)

            # absolute L2 per sample
            l2_batch = torch.sqrt((flat_diff**2).sum(dim=1))       # (B,)
            # exact-solution L2 per sample
            norm_exact = torch.sqrt((flat_exact**2).sum(dim=1))   # (B,)
            # relative L2 = abs / exact
            rel_l2_batch = l2_batch / (norm_exact +1e-8)                 # (B,)
            l1_batch = flat_diff.abs().sum(dim=1)                  # (B,)
            l1_exact = flat_exact.abs().sum(dim=1)                 # (B,)
            rel_l1_batch = l1_batch / (l1_exact + 1e-8)            # avoid divide-by-zero


            # store
            l2_errors[start:end]     = l2_batch
            rel_l2_errors[start:end] = rel_l2_batch

            l1_errors[start:end]     = l1_batch
            rel_l1_errors[start:end] = rel_l1_batch
        mean_abs_l2 = l2_errors.mean().item()
        mean_rel_l2 = rel_l2_errors.mean().item()
        mean_abs_l1 = l1_errors.mean().item()
        mean_rel_l1 = rel_l1_errors.mean().item()

        return mean_abs_l1,  mean_rel_l1, mean_abs_l2,mean_rel_l2













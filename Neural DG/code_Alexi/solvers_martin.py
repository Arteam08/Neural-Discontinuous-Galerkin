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

    left_polynomials_value = torch.tensor([1, -1], device=device).repeat(n_poly// 2 + 1)[n_poly].unsqueeze(0).unsqueeze(-1)
    right_polynomials_value = torch.ones_like(left_polynomials_value)
    return polynomials, polynomials_prime, mass_matrix, mass_matrix_inv, weights_leggauss, left_polynomials_value, right_polynomials_value





### define the solver class
class DG_solver():

    def __init__(self,t_max ,# time steps # time step
              dx, # space step
              dt,
              n_cells, # number of cells
              flow_func, # flow function, exact arguments to be clarified
              points_per_cell=40, # number of points per cell, can be replaced by a quadrature for less space storage
              n_poly=2,
              device='cpu',
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

    
    def compute_L(self,solution_DG, t, a):
        n_ic = solution_DG.shape[0]
        left_boundaries = solution_DG[:, self.left_boundary_indexes, t+a-1]
        right_boundaries = solution_DG[:, self.right_boundary_indexes, t+a-1]

        fluxes = self.flow_func(left_boundaries, right_boundaries) # à modifier pour accpeter des formes plus générales

        # !! Right polynomial values not implemented
        self.left_polynomials_value=torch.tensor([1, -1], device=self.device).repeat(self.n_poly// 2 + 1)[:self.n_poly].unsqueeze(0).unsqueeze(-1)
        fluxes = fluxes[:, 1:].unsqueeze(1) - fluxes[:, :-1].unsqueeze(1) * self.left_polynomials_value
                        
        fluxes = (self.mass_matrix_inv * fluxes)
        # plt.plot(fluxes[0, 1,:].cpu().detach().numpy())
        # plt.show()
                        
        f_u = self.f_PDE(solution_DG[:, :, t+a-1])

            
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
        L=-fluxes+residual
        return L
    
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


        print("mass_matrix_inv", self.mass_matrix_inv.shape)
        print("quadrature_weights", self.quadrature_weights.shape)
        print("basis_func", self.basis_func.shape)
        print("interp_ic", interp_ic.shape)
        print("solution_DG", solution_DG.shape)
        temp = self.mass_matrix_inv * self.quadrature_weights * self.basis_func  
        # temp has shape [1, n_poly, points_per_cell]; squeeze the singleton batch dimension:
        temp = temp.squeeze(0)  # now shape [n_poly, points_per_cell]

        weights_dg = torch.einsum('lp,icp->ilc', temp, interp_ic) * (self.dx / 2)

        






        reconstructed_solution = torch.einsum('ilc,lk->ick', weights_dg, self.basis_func.squeeze(0))
        reconstructed_solution = reconstructed_solution.reshape(n_ic, self.n_cells * self.points_per_cell)
        solution_DG[:, :, 0] = reconstructed_solution



        return weights_dg, solution_DG, n_ic
 
    def solve(self, initial_conditions):
        weights_dg, solution_DG, n_ic = self.process_initial_conditions(initial_conditions)
        for t in range(1, self.t_max):
            weights_dg_save = weights_dg.clone() #Ut
            for a in range(2):
                #RK2
                #U{t-1} si a =0, U{t} si a=1
                L=self.compute_L(solution_DG, t, a)
        
                weights_dg = weights_dg_save + (self.dt*L).squeeze(1) * (1/2 if a==0 else 1.) #weights_dg is not saved but used in the solution

                solution_DG[:, :, t] = torch.einsum('ijk,ijl->ikl', self.basis_func, weights_dg).permute(0, 2, 1).reshape(n_ic, -1)
        return solution_DG





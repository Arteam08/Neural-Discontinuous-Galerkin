import argparse
import torch
import os
import hashlib
from solvers_martin import DG_solver

import numpy as np 
import torch.nn.functional as F

import nn_arz.nn_arz.utils
import nn_arz.nn_arz.lwr.explicit
import nn_arz.nn_arz.lwr.fluxes
import nn_arz.nn_arz.lwr.lax_hopf
import nn_arz.nn_arz.polynomials

import time
import matplotlib.pyplot as plt

save_folder = 'data_martin'

#parameters for genration
N = 4
device = 'cpu'
values_for_complex = 10

#parameters of the problem
dx= 1e-2
n_cells = 100
t_max = 600
dt=1e-4
n_poly = 2
points_per_cell = 20
flux_function = nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield

def cell_averaging(solution_dg, n_cells, points_per_cell):
    #solution_dg is of shape (n_cells, n_cells*points_per_cell, t_max), we want to average over the points_per_cell
    ic, T= solution_dg.shape[0], solution_dg.shape[2]
    solution_dg = solution_dg.reshape(ic, n_cells, points_per_cell, T)
    solution_dg = solution_dg.mean(dim=2)
    return solution_dg

solver = DG_solver(
    t_max=t_max,
    dt=dt,
    dx=dx,
    n_cells=n_cells,
    flow_func=flux_function,
    points_per_cell=points_per_cell,
    n_poly=n_poly,
    device=device
)

n_points = n_cells# * args.points_per_cell
x_max = dx * n_points / 2
x = torch.linspace(-x_max, x_max, n_points, device=device)
time = torch.arange(1, t_max + 1, device=device).unsqueeze(0) * dt

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

solutions_riemann = nn_arz.nn_arz.lwr.explicit.greenshieldRiemannSolution(x, time, c1, c2) #true solution to Riemann
####computation of the complex solution
ratio=10 # nombre de points pour calculer la moyenne dans l'intervalle avel lax_hopf=quadrature
batch_size_complex = 1


### complex ic
complex_ic = 4*torch.rand((N*(N-1), values_for_complex), device=device)
solution_complex = torch.empty((complex_ic.shape[0], n_points,t_max))
for i in range(complex_ic.shape[0]//batch_size_complex):
    temp = nn_arz.nn_arz.lwr.lax_hopf.Lax_Hopf_solver_Greenshield(
        complex_ic[i*batch_size_complex:(i+1)*batch_size_complex], 
        dx=dx/ratio, 
        dt=dt, 
        Nx=n_points*ratio, 
        Nt=t_max, 
        device=device
    )
    solution_complex[i*batch_size_complex:(i+1)*batch_size_complex] = temp.reshape((
        temp.shape[0], 
        temp.shape[1], 
        temp.shape[2] // ratio, 
        ratio)
    ).mean(dim=3).transpose(1, 2)





#ic shoulf be of shape (n_ic, some discretization)
riemmann_ic_tensor=result = torch.stack(riemann_ic).transpose(0,1)
n_ic = riemmann_ic_tensor.shape[0]

print("tensor ic",riemmann_ic_tensor.shape)
ic_reduced= riemmann_ic_tensor[:1]
print("ic_reduced",ic_reduced.shape)
print(ic_reduced)
solution_DG= solver.solve(complex_ic) 
averaged_solution_DG = cell_averaging(solution_DG, n_cells, points_per_cell)

print("solution_DG",solution_DG.shape)
print("solution_riemmann",solutions_riemann.squeeze().shape)

plot_ic = False
ic=100
if plot_ic:
    x= torch.linspace(-x_max, x_max, n_cells*points_per_cell, device=device)
    plt.scatter(x,solution_DG[ic,:,0].cpu().T, label='DG')
    plt.scatter(x,solution_DG[ic,:,1].cpu().T, label='DG')
    plt.show()
else:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(solution_complex.squeeze()[0].cpu().T, cmap='viridis', aspect='auto')
    axes[1].imshow(averaged_solution_DG[0].cpu().T, cmap='viridis', aspect='auto')
    plt.show()

# plt.imshow(solutions_riemann.squeeze()[0].cpu().T, cmap='viridis', aspect='auto')
# plt.imshow(solution_DG[0].cpu().T, cmap='viridis', aspect='auto')
# plt.show()
# print("solution_DG",solution_DG[0])


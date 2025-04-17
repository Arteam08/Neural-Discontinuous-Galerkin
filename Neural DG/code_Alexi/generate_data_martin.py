import os
import torch
import numpy as np
from solvers_martin import DG_solver
import nn_arz.nn_arz.lwr.explicit as expl
import nn_arz.nn_arz.lwr.fluxes as fluxes
import nn_arz.nn_arz.lwr.lax_hopf


def generate_riemann_dataset(
    save_folder: str,
    filename: str = "riemann_dataset.pt",
    ic_boundaries: np.ndarray =[0, 4],
    N_ic_range=14,
    dx: float = 1e-2,
    n_cells: int = 20,
    points_per_cell: int = 10,
    n_poly: int = 2,
    t_max: int = 600,
    dt: float = 1e-4,
    device: str = "cpu",
):
    """
    Generate all Riemann ICs, compute exact & DG‐averaged solutions, and save to disk.

    - save_folder: directory where file will be written (created if needed)
    - filename:     name of the .pt file
    - ic_range:     1D array of values c in [0, ...], initial states
    - rest:         solver parameters (dx, n_cells, dt, etc.)

    The output file contains a dict with:
      'ic'        : Tensor[n_ic, 2]         initial jumps (c_left, c_right)
      'sol_exact' : Tensor[n_ic, n_cells, t_max]   GreenshieldRiemannSolution
      'x'         : Tensor[n_cells]         spatial grid
      'time'      : Tensor[t_max]           time grid
    """
    # make sure folder exists
    os.makedirs(save_folder, exist_ok=True)
    ic_range= np.linspace(ic_boundaries[0], ic_boundaries[1], N_ic_range)
    # build all pairs (c1, c2), excluding c1==c2
    points = np.array([[c1, c2]
                       for c1 in ic_range for c2 in ic_range
                       if c1 != c2], dtype=np.float32)
    n_ic = len(points)

    # pack into tensors
    c1 = torch.from_numpy(points[:, 0]).to(device)
    c2 = torch.from_numpy(points[:, 1]).to(device)
    ic_tensor = torch.stack([c1, c2], dim=1)  # shape (n_ic, 2)

    # build grids
    x_max = dx * n_cells / 2
    x = torch.linspace(-x_max, x_max, n_cells, device=device)
    time = torch.arange(1, t_max+1, device=device) * dt

    # exact (Riemann) solution
    sol_exact = expl.greenshieldRiemannSolution(x, time.unsqueeze(0), c1, c2)
    # shape: (1, n_ic, n_cells, t_max) → squeeze to (n_ic, n_cells, t_max)
    sol_exact = sol_exact.squeeze(0).transpose(0,1)  # reorder if needed


    # assemble and save
    dataset = {
        "ic":        ic_tensor,   # (n_ic,2)
        "sol_exact": sol_exact,   # (n_ic,n_cells,t_max)
        "x":         x,           # (n_cells,)
        "time":      time,        # (t_max,)
    }
    path = os.path.join(save_folder, filename)
    torch.save(dataset, path)
    print(f"Saved dataset with {n_ic} samples to {path}")

def generate_piecewise_constant_dataset(
    save_folder: str,
    filename: str = "piecewise_dataset.pt",
    ic_boundaries: np.ndarray =[0, 4],
    N_ic_range=14,
    dx: float = 1e-2,
    n_cells: int = 20,
    points_per_cell: int = 10,
    n_poly: int = 2,
    t_max: int = 600,
    dt: float = 1e-4,
    batch_size: int = 3, # number of piecewise constant functions computed at once
    ratio: int = 10, # number of points in the scheme to estimate the value on a cell
    device: str = "cpu",
):
    """
    Generate all Riemann ICs, compute exact & DG‐averaged solutions, and save to disk.

    - save_folder: directory where file will be written (created if needed)
    - filename:     name of the .pt file
    - ic_range:     1D array of values c in [0, ...], initial states
    - rest:         solver parameters (dx, n_cells, dt, etc.)

    The output file contains a dict with:
      'ic'        : Tensor[n_ic, 2]         initial jumps (c_left, c_right)
      'sol_exact' : Tensor[n_ic, n_cells, t_max]   GreenshieldRiemannSolution
      'x'         : Tensor[n_cells]         spatial grid
      'time'      : Tensor[t_max]           time grid
    """
    # make sure folder exists
    os.makedirs(save_folder, exist_ok=True)
    ic_range= np.linspace(ic_boundaries[0], ic_boundaries[1], N_ic_range)
    # build all pairs (c1, c2), excluding c1==c2
    points = np.array([[c1, c2]
                       for c1 in ic_range for c2 in ic_range
                       if c1 != c2], dtype=np.float32)
    n_ic = len(points)

    # pack into tensors
    c1 = torch.from_numpy(points[:, 0]).to(device)
    c2 = torch.from_numpy(points[:, 1]).to(device)
    ic_tensor = torch.stack([c1, c2], dim=1)  # shape (n_ic, 2)

    # build grids
    x_max = dx * n_cells / 2
    x = torch.linspace(-x_max, x_max, n_cells, device=device)
    time = torch.arange(1, t_max+1, device=device) * dt

    # exact (Riemann) solution
    solution_exact = torch.empty((ic_tensor.shape[0], n_cells,t_max))
    for i in range(ic_tensor.shape[0]//batch_size):
        temp = nn_arz.nn_arz.lwr.lax_hopf.Lax_Hopf_solver_Greenshield(
            ic_tensor[i*batch_size:(i+1)*batch_size], 
            dx=dx/ratio, 
            dt=dt, 
            Nx=n_cells*ratio, 
            Nt=t_max, 
            device=device
        )
        solution_exact[i*batch_size:(i+1)*batch_size] = temp.reshape((
            temp.shape[0], 
            temp.shape[1], 
            temp.shape[2] // ratio, 
            ratio)
        ).mean(dim=3).transpose(1, 2)
    # shape: (1, n_ic, n_cells, t_max) → squeeze to (n_ic, n_cells, t_max)
    sol_exact = sol_exact.squeeze(0).transpose(0,1)  # reorder if needed


    # assemble and save
    dataset = {
        "ic":        ic_tensor,   # (n_ic,2)
        "sol_exact": sol_exact,   # (n_ic,n_cells,t_max)
        "x":         x,           # (n_cells,)
        "time":      time,        # (t_max,)
    }
    path = os.path.join(save_folder, filename)
    torch.save(dataset, path)
    print(f"Saved dataset with {n_ic} samples to {path}")


class HyperbolicDataset(Dataset):
    def __init__(self, path):
        """
        Expects a .pt file containing a dict with keys:
          'ic'        : Tensor[N, 2]
          'sol_exact' : Tensor[N, C, T]
          'sol_dg'    : Tensor[N, C, T]
          'x'         : Tensor[C]
          'time'      : Tensor[T]
        """
        data = torch.load(path)
        self.ic        = data['ic']          # (N,2)
        self.sol_exact = data['sol_exact']   # (N,C,T)
        self.sol_dg    = data['sol_dg']      # (N,C,T)

    def __len__(self):
        return self.ic.shape[0]

    def __getitem__(self, idx):
        # here you can choose what your "input" and "target" are
        # e.g. input = IC, target = exact solution at all times
        return {
            'ic':        self.ic[idx],         # (2,)
            'sol_exact': self.sol_exact[idx],  # (C,T)
            'sol_dg':    self.sol_dg[idx],     # (C,T)
        }


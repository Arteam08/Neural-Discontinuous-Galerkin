import os
import torch
import numpy as np
import nn_arz.nn_arz.lwr.explicit as expl
import nn_arz.nn_arz.lwr.fluxes as fluxes
import nn_arz.nn_arz.lwr.lax_hopf
from torch.utils.data import Dataset, DataLoader



def generate_riemann_dataset(
    save_folder: str,
    filename: str = "riemann_dataset.pt",
    ic_boundaries: np.ndarray = [0, 4],
    N_ic_range: int = 14,
    dx: float = 1e-2,
    n_cells: int = 20,
    t_max: int = 600,
    dt: float = 1e-4,
    device: str = "cpu",
    batch_size: int = 64,
):
    """
    Generate all Riemann ICs in batches, compute exact solutions, and save to disk.

    - save_folder: directory where file will be written
    - filename:     name of the .pt file
    - rest:         solver parameters

    Outputs:
      'ic'        : Tensor[n_ic, 2]
      'sol_exact' : Tensor[n_ic, n_cells, t_max]
      'x'         : Tensor[n_cells]
      'time'      : Tensor[t_max]
    """
    os.makedirs(save_folder, exist_ok=True)

    ic_range = np.linspace(ic_boundaries[0], ic_boundaries[1], N_ic_range)
    points = np.array([[c1, c2]
                       for c1 in ic_range for c2 in ic_range
                       if c1 != c2], dtype=np.float32)
    n_ic = len(points)

    # (n_ic, 2)
    c1 = torch.from_numpy(points[:, 0]).to(device)
    c2 = torch.from_numpy(points[:, 1]).to(device)
    ic_tensor = torch.stack([c1, c2], dim=1)  # (n_ic, 2)

    # (n_cells,)
    x = torch.linspace(-dx*n_cells/2, dx*n_cells/2, n_cells, device=device)  # (n_cells,)
    # (t_max,)
    time = torch.arange(1, t_max+1, device=device) * dt  # (t_max,)

    # (n_ic, n_cells, t_max)
    sol_exact = torch.empty((n_ic, n_cells, t_max), device=device)  # (n_ic, n_cells, t_max)

    for i in range((n_ic + batch_size - 1) // batch_size):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_ic)
        batch_c1 = c1[start:end]
        batch_c2 = c2[start:end]

        # returns (1, batch, n_cells, t_max)
        batch_out = expl.greenshieldRiemannSolution(
            x, time.unsqueeze(0), batch_c1, batch_c2
        )
        # remove leading dim → (batch, n_cells, t_max)
        batch_sol = batch_out.squeeze(0)  # (batch, n_cells, t_max)

        sol_exact[start:end] = batch_sol

    dataset = {
        "ic":        ic_tensor,   # (n_ic, 2)
        "sol_exact": sol_exact,   # (n_ic, n_cells, t_max)
        "x":         x,           # (n_cells,)
        "time":      time,        # (t_max,)
    }
    path = os.path.join(save_folder, filename)
    torch.save(dataset, path)
    print(f"Saved dataset with {n_ic} samples to {path}",f"Saved dataset with {n_ic} samples to {path}")


# à comprendre en détails
def generate_piecewise_constant_dataset(
    save_folder: str,
    filename: str = "piecewise_dataset.pt",
    N_pieces=10,
    amplitude: float = 4.,
    n_ic=20,
    dx: float = 1e-2,
    n_cells: int = 20,
    points_per_cell: int = 10,
    t_max: int = 600,
    dt: float = 1e-4,
    batch_size: int = 3,  # (batch,)
    ratio: int = 10,
    device: str = "cpu",
):
    """
    Generate piecewise constant ICs, compute Riemann & DG‐averaged solutions in batches, and save to disk.

    Outputs:
      'ic'        : Tensor[n_ic, 2]
      'sol_exact' : Tensor[n_ic, n_cells, t_max]
      'x'         : Tensor[n_cells]
      'time'      : Tensor[t_max]
    """
    os.makedirs(save_folder, exist_ok=True)

    ic_tensor = amplitude*torch.rand(n_ic, N_pieces, device=device)

    # (n_cells,)
    x = torch.linspace(-dx * n_cells/2, dx * n_cells/2, n_cells, device=device)  # (n_cells,)
    # (t_max,)
    time = torch.arange(1, t_max+1, device=device) * dt  # (t_max,)

    # allocate exact solution array (n_ic, n_cells, t_max)
    sol_exact = torch.empty((n_ic, n_cells, t_max), device=device)  # (n_ic, space, time)

    for i in range((n_ic + batch_size - 1) // batch_size):
        start = i * batch_size
        end   = min((i + 1) * batch_size, n_ic)
        batch_ic = ic_tensor[start:end]  # (batch, space)

        # solver returns (batch, time, space*ratio)
        temp = nn_arz.nn_arz.lwr.lax_hopf.Lax_Hopf_solver_Greenshield(
            batch_ic,
            dx=dx/ratio,
            dt=dt,
            Nx=n_cells * ratio,
            Nt=t_max,
            device=device
        )  # temp: (B, T, X*R)

        # reshape time‑major → (B, T, X, R)
        temp = temp.reshape(
            temp.shape[0],        # B
            temp.shape[1],        # T
            temp.shape[2] // ratio,  # X
            ratio                 # R
        )  # (B, T, space, ratio)

        # average over the 'ratio' axis → (B, T, space)
        temp = temp.mean(dim=3)  # (B, T, space)

        # swap axes to get (B, space, T)
        batch_sol = temp.transpose(1, 2)  # (B, space, time)

        sol_exact[start:end] = batch_sol  # write into (n_ic, space, time)

    dataset = {
        "ic":        ic_tensor,   # (n_ic, 2)
        "sol_exact": sol_exact,   # (n_ic, n_cells, t_max)
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
          'x'         : Tensor[C]
          'time'      : Tensor[T]
        """
        data = torch.load(path)
        self.ic        = data['ic']          # (N,2)
        self.sol_exact = data['sol_exact']   # (N,C,T)

    def __len__(self):
        return self.ic.shape[0]

    def __getitem__(self, idx):
        # here you can choose what your "input" and "target" are
        # e.g. input = IC, target = exact solution at all times
        return {
            'ic':        self.ic[idx],         # (2,)
            'sol_exact': self.sol_exact[idx],  # (C,T)
        }


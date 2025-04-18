# define on func to run the solver and save the results and make the plots
# define one func to 

import matplotlib.pyplot as plt
from solvers_martin import DG_solver
import torch
from generate_data_martin import HyperbolicDataset
from torch.utils.data import Dataset, DataLoader


import os

def plot_solver(solver, dataset_path, batch_size, plot_path):
    """
    Run the solver, compute per-sample absolute & relative L2 errors in one pass,
    and output comparison heatmaps for the first 3 ICs.
    """
    device = solver.device

    # Load dataset
    ds = HyperbolicDataset(dataset_path)
    N = len(ds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Pre-allocate error tensors
    l2_errors      = torch.empty(N, device=device)  # (N,)
    rel_l2_errors  = torch.empty(N, device=device)  # (N,)

    # Batch-wise solve & error
    for batch_idx, batch in enumerate(loader):
        start = batch_idx * batch_size
        end   = start + batch['ic'].shape[0]

        ic         = batch['ic'].to(device)         # (B, 2)
        sol_exact  = batch['sol_exact'].to(device)  # (B, C, T)

        # DG solve + averaging → (B, C, T)
        sol_DG_large = solver.solve(ic)
        sol_DG       = solver.cell_averaging(sol_DG_large)

        # flatten spatial/time dims
        diff = sol_DG - sol_exact                  # (B, C, T)
        flat_diff = diff.view(diff.shape[0], -1)   # (B, C*T)
        flat_exact = sol_exact.view(sol_exact.shape[0], -1)  # (B, C*T)

        # absolute L2 per sample
        l2_batch = torch.sqrt((flat_diff**2).sum(dim=1))       # (B,)
        # exact-solution L2 per sample
        norm_exact = torch.sqrt((flat_exact**2).sum(dim=1))   # (B,)
        # relative L2 = abs / exact
        rel_l2_batch = l2_batch / norm_exact                  # (B,)

        # store
        l2_errors[start:end]     = l2_batch
        rel_l2_errors[start:end] = rel_l2_batch

    # --- Plot heatmaps for the first 3 ICs ---
    full_data = torch.load(dataset_path)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for i in range(3):
        exact = full_data['sol_exact'][i].cpu().numpy()  # (C, T)
        dg    = solver.cell_averaging(
                    solver.solve(full_data['ic'][i:i+1].to(device))
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
    mean_abs = l2_errors.mean().item()
    mean_rel = rel_l2_errors.mean().item()
    print(f"Saved comparison heatmaps to {plot_path}")
    print(f"Mean absolute L2 error: {mean_abs:.4e}")
    print(f"Mean relative L2 error: {mean_rel:.4e}")

def run_solver(solver, dataset_path, batch_size):
    """outputs the solution tensor"""
    ds = HyperbolicDataset(dataset_path)
    N = len(ds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for batch_idx, batch in enumerate(loader):
        start = batch_idx * batch_size
        end   = start + batch['ic'].shape[0]

        ic         = batch['ic'].to(solver.device)         # (B, 2)
        sol_exact  = batch['sol_exact'].to(solver.device)  # (B, C, T)

        # DG solve + averaging → (B, C, T)
        sol_DG_large = solver.solve(ic)
        sol_DG       = solver.cell_averaging(sol_DG_large)
        return sol_DG



# define on func to run the solver and save the results and make the plots
# define one func to 

import matplotlib.pyplot as plt
from solvers_martin import DG_solver
import torch
from generate_data_martin import HyperbolicDataset
from torch.utils.data import Dataset, DataLoader


import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def run_solver(solver, dataset_path, batch_size,  plot_path):
    """
    Run the solver, compute per-sample L2 error in one pass (no list),
    save results, and output comparison heatmaps for the first 3 ICs.
    """
    device=solver.device
    # Load dataset
    ds = HyperbolicDataset(dataset_path)
    N = len(ds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Pre-allocate error tensor
    l2_errors = torch.empty(N, device=device)

    # Batch-wise solve & error
    for batch_idx, batch in enumerate(loader):
        start = batch_idx * batch_size
        end   = start + batch['ic'].shape[0]

        ic        = batch['ic'].to(device)         # (B, 2)
        sol_exact = batch['sol_exact'].to(device)  # (B, C, T)

        # DG solve + average → (B, C, T)
        sol_DG_large = solver.solve(ic)
        sol_DG       = solver.cell_averaging(sol_DG_large)

        # Compute L2 norm for each sample and store
        diff = sol_DG - sol_exact
        l2_batch = torch.sqrt((diff**2).view(diff.shape[0], -1).sum(dim=1))
        l2_errors[start:end] = l2_batch



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
    print(f"Saved comparison heatmaps to {plot_path}")
    print("l2_error", l2_errors.mean())

        



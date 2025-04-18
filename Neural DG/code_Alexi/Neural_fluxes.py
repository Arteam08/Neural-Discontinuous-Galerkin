## we define the classes for neural fluxes 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from generate_data_martin import *
from eval_and_plot_funcs_martin import *
from solvers_martin import DG_solver

class MLPFlux_2_value(nn.Module):
    """MLP wich commputes the flux from interface values only"""
    def __init__(self,device, hidden_dims=[32,32], activation=F.relu):
        super().__init__()
        dims = [2] + hidden_dims + [1]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        )
        self.activation = activation
        self._init_weights()
        self.device = device

    def _init_weights(self):
        # Xavier‐uniform for all Linear layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)



    def forward(self, uL, uR):
        """
        uL, uR: each is (batch, n_cells)
        returns flux: (batch, n_cells)
        """
        # stack along last dim → (batch, n_cells, 2)
        x = torch.stack([uL, uR], dim=-1)

        # collapse batch & cell dims → (batch*n_cells, 2)
        B, N = x.shape[0], x.shape[1]
        x = x.view(B*N, 2)

        # run through MLP
        for lin in self.layers[:-1]:
            x = self.activation(lin(x))
        x = self.layers[-1](x)  # final linear

        # restore shape → (batch, n_cells)
        flux = x.view(B, N)
        return flux
    


    def train_to_func(self,
                    flow_func,
                    lr: float = 1e-3,
                    n_epochs: int = 1000,
                    loss_fn=nn.MSELoss(),
                    batch_size: int = 1000,
                    u_amplitude: float = 2):
        """
        Pretrains the NN to match the explicit flux function, then
        plots NN vs ground‑truth flux + abs diff + loss curve.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        # ---- Training loop ----
        for epoch in range(n_epochs):
            # sample random interface states (batch,1)
            uL = torch.rand(batch_size, 1, device=self.device) * u_amplitude
            uR = torch.rand(batch_size, 1, device=self.device) * u_amplitude

            # forward + loss
            flux_pred = self.forward(uL, uR)                  # (batch,1)
            with torch.no_grad():
                flux_true = flow_func(uL, uR)                 # (batch,1)
            loss = loss_fn(flux_pred, flux_true)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch:4d}/{n_epochs}  Loss: {loss.item():.4e}')

        print("Training finished")

        # ---- Build evaluation grid ----
        us = torch.linspace(-u_amplitude, u_amplitude, 100, device=self.device)
        uLg, uRg = torch.meshgrid(us, us, indexing="ij")    # (100,100)
        uLf = uLg.reshape(-1, 1)                            # (10000,1)
        uRf = uRg.reshape(-1, 1)                            # (10000,1)

        with torch.no_grad():
            f_nn_flat = self.forward(uLf, uRf)              # (10000,1)
            f_gt_flat = flow_func(uLf, uRf)                 # (10000,1)

        f_nn   = f_nn_flat.view(100, 100).cpu().numpy()     # (100,100)
        f_gt   = f_gt_flat.view(100, 100).cpu().numpy()     # (100,100)
        f_diff = np.abs(f_nn - f_gt)                        # (100,100)

        # shared color‐scale for NN vs GT
        vmin = min(f_nn.min(), f_gt.min())
        vmax = max(f_nn.max(), f_gt.max())
        extent = [-u_amplitude, u_amplitude, -u_amplitude, u_amplitude]

        # ---- Plot everything in 2×2 ----
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # [0,0] NN-predicted flux
        im0 = axes[0,0].imshow(f_nn, origin="lower", aspect="auto",
                            extent=extent, vmin=vmin, vmax=vmax)
        axes[0,0].set(title="NN‑predicted flux", xlabel="u_R", ylabel="u_L")
        fig.colorbar(im0, ax=axes[0,0])

        # [0,1] Ground-truth flux
        im1 = axes[0,1].imshow(f_gt, origin="lower", aspect="auto",
                            extent=extent, vmin=vmin, vmax=vmax)
        axes[0,1].set(title="Ground‑truth flux", xlabel="u_R", ylabel="u_L")
        fig.colorbar(im1, ax=axes[0,1])

        # [1,0] Absolute difference
        im2 = axes[1,0].imshow(f_diff, origin="lower", aspect="auto",
                            extent=extent)
        axes[1,0].set(title="|NN – GT|", xlabel="u_R", ylabel="u_L")
        fig.colorbar(im2, ax=axes[1,0])

        # [1,1] Loss evolution
        axes[1,1].plot(range(n_epochs), losses)
        axes[1,1].set_yscale('log')
        axes[1,1].set(title="Training loss evolution",
                    xlabel="Epoch", ylabel="MSE Loss")

        plt.tight_layout()
        plt.show()

    def train_on_data(
    self,
    dataset_path,
    solver_params,
    batch_size: int = 1000,
    n_epochs: int   = 1000,
    lr: float       = 1e-3,
    loss_fn         = nn.MSELoss(),
    ):
        """
        Trains the model using the DG solver.
        solver_params should be a dict with all DG_solver __init__ args
        except `flow_func` (we’ll inject our network there).
        """
        # -- Dataset & loader --
        ds     = HyperbolicDataset(dataset_path)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # -- Optimizer & loss history --
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        # -- Prepare solver parameters --
        sp = solver_params.copy()
        sp['flow_func'] = self.forward
        sp['device']    = self.device
        solver = DG_solver(**sp)

        # -- Training loop --
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_rel_error = 0.0

            for batch in loader:
                ic        = batch['ic'].to(self.device)         # (B, ...)
                sol_exact = batch['sol_exact'].to(self.device)  # (B, C, T)

                # Solve & average
                sol_DG_large = solver.solve(ic)
                sol_DG       = solver.cell_averaging(sol_DG_large)

                # Absolute MSE loss
                loss = loss_fn(sol_DG, sol_exact)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Relative L2 error for this batch
                B = sol_exact.shape[0]
                diff = (sol_DG - sol_exact).view(B, -1)
                exact_flat = sol_exact.view(B, -1)
                l2_diff = torch.norm(diff, dim=1)             # (B,)
                l2_exact = torch.norm(exact_flat, dim=1)      # (B,)
                rel_batch = (l2_diff / l2_exact).mean().item()
                epoch_rel_error += rel_batch

            # end of one epoch
            avg_loss = epoch_loss / len(loader)
            avg_rel_error = epoch_rel_error / len(loader)
            losses.append(avg_loss)

            if epoch % 50 == 0 or epoch == n_epochs - 1:
                print(
                    f"Epoch {epoch:4d}/{n_epochs} - "
                    f"Loss: {avg_loss:.4e} - "
                    f"RelErr: {avg_rel_error:.4e}"
                )

        return losses
    
    def save(self, filepath: str, save_name: str):
        """
        Save this model’s parameters (state_dict) to the given filepath,
        using `save_name` as the filename. Automatically creates parent directories if needed.

        Example:
            model.save("models/checkpoints", "flux_model.pth")
            → will save to models/checkpoints/flux_model.pth
        """
        # Ensure the directory exists
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)

        full_path = os.path.join(filepath, save_name)
        torch.save(self.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(cls, filepath: str):
        """
        Load a saved MLPFlux_2_value model from the given path.

        Args:
            filepath (str): Path to the saved .pth file.
            device (str): Device to map the model to (default: 'cpu').

        Returns:
            An instance of MLPFlux_2_value with loaded weights.
        """
        # Create model instance with correct device
        model = cls(device=self.device)
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {filepath}")
        return model



 













class CNNFlux(nn.Module):
    """
    CNN that learns the interface flux from u_L and u_R.
    Input:  uL, uR each of shape (batch, n_cells)
    Output: flux of shape (batch, n_cells)
    """
    def __init__(self, device, channels=[16,32], kernel_size=3):
        """
        channels: list of hidden channel sizes for conv layers
        kernel_size: 1D convolution kernel width (must be odd for same padding)
        """
        super().__init__()
        self.device = device

        # assemble conv1d layers: in_channels=2 -> channels[0] -> channels[1] -> 1
        convs = []
        in_ch = 2
        pad = kernel_size // 2
        for out_ch in channels:
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad))
            in_ch = out_ch
        # final 1×1 conv to collapse to 1 channel
        convs.append(nn.Conv1d(in_ch, 1, kernel_size=1))
        self.convs = nn.ModuleList(convs)

        # initialize
        for m in self.convs:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, uL, uR):
        """
        uL, uR: each (batch, n_cells)
        returns flux: (batch, n_cells)
        """
        # stack into channels → (batch, 2, n_cells)
        x = torch.stack([uL, uR], dim=1).to(self.device)

        # run through conv layers with ReLU except last
        for conv in self.convs[:-1]:
            x = F.relu(conv(x))
        x = self.convs[-1](x)         # → (batch, 1, n_cells)

        return x.squeeze(1)           # → (batch, n_cells)


    def train_to_func(self,
                      flow_func,
                      lr: float = 1e-3,
                      n_epochs: int = 1000,
                      loss_fn=nn.MSELoss(),
                      batch_size: int = 1000,
                      u_amplitude: float = 2):
        """
        Pretrains the CNN to match the explicit flux, then
        plots NN vs ground‑truth flux + abs diff + loss curve.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        # ---- Training loop ----
        for epoch in range(n_epochs):
            uL = torch.rand(batch_size, 1, device=self.device) * u_amplitude
            uR = torch.rand(batch_size, 1, device=self.device) * u_amplitude

            pred = self.forward(uL, uR)               # (batch, n_cells)
            with torch.no_grad():
                true = flow_func(uL, uR)              # (batch, n_cells)
            loss = loss_fn(pred, true)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch:4d}/{n_epochs}  Loss: {loss:.4e}')

        print("Training finished")

        # ---- Build evaluation grid ----
        us = torch.linspace(-u_amplitude, u_amplitude, 100, device=self.device)
        uLg, uRg = torch.meshgrid(us, us, indexing="ij")    # (100,100)
        uLf = uLg.reshape(-1, 1)
        uRf = uRg.reshape(-1, 1)

        with torch.no_grad():
            f_nn_flat = self.forward(uLf, uRf)              # (10000, n_cells)
            f_gt_flat = flow_func(uLf, uRf)                 # (10000, n_cells)

        # for visualization pick one cell index (e.g. center) or plot as 2D over uL/uR if you treat n_cells=1
        # Here we assume n_cells=1 for interface flux, so:
        f_nn = f_nn_flat.view(100, 100).cpu().numpy()
        f_gt = f_gt_flat.view(100, 100).cpu().numpy()
        f_diff = np.abs(f_nn - f_gt)

        # shared color scale
        vmin, vmax = min(f_nn.min(), f_gt.min()), max(f_nn.max(), f_gt.max())
        extent = [-u_amplitude, u_amplitude, -u_amplitude, u_amplitude]

        # ---- Plot 2×2 ----
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        im0 = axes[0,0].imshow(f_nn, origin="lower", aspect="auto",
                               extent=extent, vmin=vmin, vmax=vmax)
        axes[0,0].set(title="CNN‑predicted flux", xlabel="u_R", ylabel="u_L")
        fig.colorbar(im0, ax=axes[0,0])

        im1 = axes[0,1].imshow(f_gt, origin="lower", aspect="auto",
                               extent=extent, vmin=vmin, vmax=vmax)
        axes[0,1].set(title="Ground‑truth flux", xlabel="u_R", ylabel="u_L")
        fig.colorbar(im1, ax=axes[0,1])

        im2 = axes[1,0].imshow(f_diff, origin="lower", aspect="auto",
                               extent=extent)
        axes[1,0].set(title="|CNN – GT|", xlabel="u_R", ylabel="u_L")
        fig.colorbar(im2, ax=axes[1,0])

        axes[1,1].plot(range(n_epochs), losses)
        axes[1,1].set_yscale('log')
        axes[1,1].set(title="Training loss evolution",
                      xlabel="Epoch", ylabel="MSE Loss")

        plt.tight_layout()
        plt.show()

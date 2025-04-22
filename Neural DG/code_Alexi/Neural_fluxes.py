import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from generate_data_martin import generate_riemann_dataset
from solvers_martin import DG_solver
from torch.utils.tensorboard import SummaryWriter

import nn_arz.nn_arz.lwr.fluxes





class BaseFluxModel(nn.Module):
    """
    Base class for flux models. Child classes must implement `forward(uL, uR)`.
    Provides common training and plotting utilities.
    """
    def __init__(self, device, *args, **kwargs):
        super().__init__()
        self._init_kwargs=dict(device=device, *(), **kwargs)        
        self._init_kwargs=dict(device=device, **kwargs)
        self.device = device

    def train_to_func(
        self,
        flow_func,
        checkpoint_folder,
        pre_train_name,
        lr: float = 1e-3,
        n_epochs: int = 1000,
        loss_fn=nn.MSELoss(),
        batch_size: int = 1000,
        u_amplitude: float = 2,
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = scheduler_class(optimizer, mode='min', patience=100, factor=0.5, verbose=True)
        losses, rel_l2s, rel_l1s = [], [], []

        for epoch in range(n_epochs):
            # Sample uniformly on [0, u_amplitude]
            uL = torch.rand(batch_size, 1, device=self.device) * u_amplitude
            uR = torch.rand(batch_size, 1, device=self.device) * u_amplitude

            flux_pred = self.forward(uL, uR)
            with torch.no_grad():
                flux_true = flow_func(uL, uR)

            loss = loss_fn(flux_pred, flux_true)
            diff = flux_pred - flux_true
            rel_l2 = (torch.norm(diff) / torch.norm(flux_true)).item()
            rel_l1 = (torch.norm(diff, p=1) / torch.norm(flux_true, p=1)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            losses.append(loss.item())
            rel_l2s.append(rel_l2)
            rel_l1s.append(rel_l1)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs} | Loss: {loss.item():.4e} | RelL2: {rel_l2:.4e} | RelL1: {rel_l1:.4e}")
                self.save(checkpoint_folder, pre_train_name)

        print("Training to function finished.")
        self._plot_pretrain(u_amplitude, flow_func, losses, rel_l2s)

    def _plot_pretrain(self, u_amplitude, flow_func, losses, rel_l2s):
        # Evaluation grid
        us = torch.linspace(0, u_amplitude, 100, device=self.device)
        uLg, uRg = torch.meshgrid(us, us, indexing='ij')
        uLf, uRf = uLg.reshape(-1, 1), uRg.reshape(-1, 1)

        with torch.no_grad():
            f_nn_flat = self.forward(uLf, uRf)
            f_gt_flat = flow_func(uLf, uRf)

        f_nn = f_nn_flat.view(100, 100).cpu().numpy()
        f_gt = f_gt_flat.view(100, 100).cpu().numpy()
        f_diff = np.abs(f_nn - f_gt)
        rel_err_grid = f_diff / (np.abs(f_gt) + 1e-8)

        extent = [0, u_amplitude, 0, u_amplitude]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        vmin, vmax = min(f_nn.min(), f_gt.min()), max(f_nn.max(), f_gt.max())

        im0 = axes[0,0].imshow(f_nn, origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
        axes[0,0].set(title='NN-predicted flux', xlabel='u_R', ylabel='u_L')
        fig.colorbar(im0, ax=axes[0,0])

        im1 = axes[0,1].imshow(f_gt, origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
        axes[0,1].set(title='Ground-truth flux', xlabel='u_R', ylabel='u_L')
        fig.colorbar(im1, ax=axes[0,1])

        im2 = axes[1,0].imshow(rel_err_grid, origin='lower', extent=extent, aspect='auto')
        axes[1,0].set(title='Relative error', xlabel='u_R', ylabel='u_L')
        fig.colorbar(im2, ax=axes[1,0])

        axes[1,1].plot(losses, label='Loss (MSE)')
        axes[1,1].plot(rel_l2s, label='Rel L2')
        axes[1,1].set_yscale('log')
        axes[1,1].set(title='Training curves', xlabel='Epoch')
        axes[1,1].legend()

        plt.tight_layout()
        plt.savefig("Neural DG/code_Alexi/plots/pretrain_diagnostics.png")
        plt.close()

    def train_consistency(self, flow_func, n_epochs, batch_size, sheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau, lr=1e-3, u_amplitude=4):
        """trains the NN to be consistent, and to be monotonous"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = sheduler_class(optimizer, mode='min', patience=100, factor=0.5, verbose=True)
        losses, rel_l2s, rel_l1s = [], [], []
        for epoch in range(n_epochs):
            # Sample uniformly on [0, u_amplitude]
            u = torch.rand(batch_size, 1, device=self.device) * u_amplitude
            f_hat_uu= self.forward(u, u)
            with torch.no_grad():
                f_u = flow_func(u, u)
        
            
            diff = f_u - f_hat_uu
            consist_loss = torch.mean((diff)**2)
            

            loss = consist_loss
            rel_l2 = (torch.norm(diff) / torch.norm(f_u)).item()
            rel_l1 = (torch.norm(diff, p=1) / torch.norm(f_u, p=1)).item()
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            losses.append(loss.item())
            rel_l2s.append(rel_l2)
            rel_l1s.append(rel_l1)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{n_epochs} | Loss: {loss.item():.4e} | RelL2: {rel_l2:.4e} | RelL1: {rel_l1:.4e}")
        print("Training to function finished.")
        self._plot_pretrain(u_amplitude, flow_func, losses, rel_l2s)



    def train_on_data(
        self,
        dataset_path,
        solver_params,
        batch_size: int = 1000,
        n_epochs: int   = 1000,
        lr: float       = 1e-3,
        loss_fn         = nn.MSELoss(),
        benchmark_flow_func=nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield,
        save_folder: str = "Neural DG/code_Alexi/checkpoints",
        save_name_prefix: str = "model_data",
        TENSOR_LOG: bool = False,
            ):
  
        os.makedirs(save_folder, exist_ok=True)
        best_loss = float('inf')
        if TENSOR_LOG:
            log_dir = os.path.join("runs", save_name_prefix)
            writer = SummaryWriter(log_dir=log_dir)
  


        # --- Benchmark at start ---
        ds_bench = __import__('generate_data_martin').HyperbolicDataset(dataset_path)
        loader_bench = DataLoader(ds_bench, batch_size=batch_size, shuffle=False)

        sp_b = solver_params.copy()
        sp_b['flow_func'] = self.forward
        sp_b['device']    = self.device
        solver = DG_solver(**sp_b)

        sp2 = solver_params.copy()
        sp2['flow_func'] = benchmark_flow_func
        sp2['device']    = self.device
        solver2 = DG_solver(**sp2)

        l1, l1_rel, l2, l2_rel       = solver.compute_metrics(loader_bench)
        l1_bis, l1_rel_bis, l2_bis, l2_rel_bis = solver2.compute_metrics(loader_bench)

        print(f"""\
        Model     | L2 Abs: {l2:.4e} | L2 Rel: {l2_rel:.4e} | L1 Abs: {l1:.4e} | L1 Rel: {l1_rel:.4e}
        Benchmark | L2 Abs: {l2_bis:.4e} | L2 Rel: {l2_rel_bis:.4e} | L1 Abs: {l1_bis:.4e} | L1 Rel: {l1_rel_bis:.4e}
        """)

        # --- Now training loop ---
        ds_train = __import__('generate_data_martin').HyperbolicDataset(dataset_path)
        loader   = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        # (we reuse `solver` from above for self.forward)
        for epoch in range(1, n_epochs+1):
            epoch_loss = 0.0
            epoch_rel_l2 = 0.0
            epoch_rel_l1 = 0.0

            for batch in loader:
                ic        = batch['ic'].to(self.device)
                sol_exact = batch['sol_exact'].to(self.device)

                sol_DG_large = solver.solve(ic)
                sol_DG       = solver.cell_averaging(sol_DG_large)

                loss = loss_fn(sol_DG, sol_exact)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                B = sol_exact.shape[0]
                diff       = (sol_DG - sol_exact).view(B, -1)
                exact_flat = sol_exact.view(B, -1)

                # L2 and relative L2
                l2_diff = torch.norm(diff, dim=1)
                l2_exact = torch.norm(exact_flat, dim=1)
                rel_l2 = (l2_diff / (l2_exact + 1e-8)).mean().item()

                # L1 and relative L1
                l1_diff = diff.abs().sum(dim=1)
                l1_exact = exact_flat.abs().sum(dim=1)
                rel_l1 = (l1_diff / (l1_exact + 1e-8)).mean().item()

                epoch_rel_l2 += rel_l2
                epoch_rel_l1 += rel_l1

            avg_loss   = epoch_loss / len(loader)
            avg_rel_l2 = epoch_rel_l2 / len(loader)
            avg_rel_l1 = epoch_rel_l1 / len(loader)
            losses.append(avg_loss)

            # overwrite the “last” checkpoint every 50 epochs
            if epoch % 50 == 0:
                self.save(save_folder, f"{save_name_prefix}_last.pt")

            # save best model when improved
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(save_folder, f"{save_name_prefix}_best.pt")

            # original per-epoch display
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(
                    f"Epoch {epoch:4d}/{n_epochs} - "
                    f"Loss: {avg_loss:.4e} - "
                    f"RelL2: {avg_rel_l2:.4e} - "
                    f"RelL1: {avg_rel_l1:.4e}"
                )
                #print the convergence metrics
                consistency_percentage, negativity_uL, positivity_uR = self.convergence_metrics(flow_func=benchmark_flow_func, amplitude=4, n_eval=100)
                print(f"Consistency: {consistency_percentage:.4e} | Negativity uL: {negativity_uL:.4e} | Positivity uR: {positivity_uR:.4e}")
                if TENSOR_LOG:
                    writer.add_scalar("Loss/train", avg_loss, epoch)
                    writer.add_scalar("RelL2/train", avg_rel_l2, epoch)
                    writer.add_scalar("RelL1/train", avg_rel_l1, epoch)
                    writer.add_scalar("Convergence/Consistency", consistency_percentage, epoch)
                    writer.add_scalar("Convergence/Negativity_uL", negativity_uL, epoch)
                    writer.add_scalar("Convergence/Positivity_uR", positivity_uR, epoch)

        if TENSOR_LOG:
            writer.close()

        return losses



    def save(self, filepath: str, filename: str):
        """
        Save both the state_dict and the init‐kwargs so we can reinstantiate later.
        """
        os.makedirs(filepath, exist_ok=True)
        full_path = os.path.join(filepath, filename)
        torch.save({
            'init_kwargs': self._init_kwargs,
            'state_dict': self.state_dict()
        }, full_path)
        # print(f"Model + metadata saved to {full_path}")

    def convergence_metrics(self, flow_func=nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield, amplitude=2, n_eval=100):
        """ Compute the consitency metrics, as well as monotoniciy metrics for the model"""
        # Unifom evalution grid, wiht identic uL and uR
        us = torch.linspace(0, amplitude, n_eval, device=self.device)
        uLg, uRg = torch.meshgrid(us, us, indexing='ij') #shape (n_eval, n_eval)
        # uLf, uRf = uLg.reshape(-1, 1), uRg.reshape(-1, 1) #shape (n_eval*n_eval, 1)
        uLf = uLg.reshape(-1, 1).requires_grad_()
        uRf = uRg.reshape(-1, 1).requires_grad_()   
        
        f_nn_flat = self.forward(uLf, uRf)
        f_gt_flat = flow_func(uLf, uRf)
        #indicates whether the model is consistent
        consistency_percentage = torch.mean((torch.abs(f_nn_flat - f_gt_flat)))/(torch.mean(torch.abs(f_gt_flat))+1e-8)

        #computer monotonicity inidicators
        grad_outputs = torch.ones_like(f_nn_flat)

        grads = torch.autograd.grad(
        outputs=f_nn_flat,
        inputs=[uLf, uRf],
        grad_outputs=grad_outputs,
        create_graph=True
        )
        dF_duL, dF_duR = grads

        negativity_uL = torch.mean((dF_duL < 0).float()).item()
        positivity_uR = torch.mean((dF_duR > 0).float()).item()
        #maybe make these relative to the max value of the gradient


        return consistency_percentage.item(), negativity_uL, positivity_uR
        



    @classmethod
    def load(cls, filepath: str, device: str = None):
        """
        Load model + metadata, reinstantiate the exact same architecture,
        then load the weights.
        If `device` is provided, it will override the saved one.
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        init_kwargs = checkpoint['init_kwargs']
        # allow override of device at load time
        if device is not None:
            init_kwargs['device'] = device

        # Recreate model with the original args
        model = cls(**init_kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(model.device)
        model.eval()
        print(f"Model loaded from {filepath} with init_kwargs={init_kwargs}")
        return model


class MLPFlux_2_value(BaseFluxModel):
    def __init__(self, device, hidden_dims=[32,32], activation=F.relu):
        super().__init__(device=device,
                hidden_dims=hidden_dims,
                activation=activation)
        dims = [2] + hidden_dims + [1]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1))
        self.activation = activation
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, uL, uR, u_max=4):
        B, N = uL.shape
        x = torch.stack([uL, uR], dim=-1).view(B*N, 2)
        for lin in self.layers[:-1]:
            x = self.activation(lin(x))
        x = self.layers[-1](x)
        x = 0.5 *(u_max+1) * (torch.tanh(x / u_max)+1)-0.5#rescales to [-0.5, 4.5]
        return x.view(B, N)


class CNNFlux(BaseFluxModel):
    def __init__(self, device, channels=[16,32], kernel_size=3, activation=F.relu):

        super().__init__(
            device=device,
            channels=channels,
            kernel_size=kernel_size,
            activation=activation
        )
        in_ch = 2
        pad = kernel_size // 2
        convs = []
        for out_ch in channels:
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad))
            in_ch = out_ch
        convs.append(nn.Conv1d(in_ch, 1, kernel_size=1))
        self.convs = nn.ModuleList(convs)
        for m in self.convs:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, uL, uR):
        x = torch.stack([uL, uR], dim=1).to(self.device)
        for conv in self.convs[:-1]:
            x = F.relu(conv(x))
        x = self.convs[-1](x)
        return x.squeeze(1)

import torch 
import os

import numpy as np

from tqdm import tqdm 
from torch.utils.data import Dataset

import nn_arz.nn_arz.numerical_schemes
import nn_arz.nn_arz.utils

class dataset(Dataset):
    def __init__(self, nt, nx, dx, dt, n_points, gamma=2.):
        self.n_points = n_points
        self.nt = nt
        self.size = n_points * (n_points - 1)

        self.rho = torch.zeros((self.size, nx, nt))
        self.v = torch.zeros((self.size, nx, nt))
        self.y = torch.zeros((self.size, nx, nt))
        self.u = torch.zeros((self.size, nx, nt, 2))
        self.flux = torch.zeros((self.size, nx, nt, 2))
        self.x = torch.linspace(-nx*dx/2, nx*dx/2, nx)
        self.t = torch.linspace(0, nt*dt, nt)

        self.path = f"datasets/dataset_riemann_{nx}_{nt}_{self.size}_numeric.pt"

        if os.path.exists(self.path):
            # Load the pre-saved tensors
            data = torch.load(self.path, weights_only=True)
            self.rho = data['rho']
            self.v = data['v']
            self.y = data['y']
            self.u = data['u']
            self.flux = data['flux']
            print("Dataset loaded from file.")
        else:
            # Generating
            rho_l, rho_r = np.linspace(0, 1, self.n_points), np.linspace(0, 1, self.n_points)
            v_l, v_r = np.linspace(0, 1, self.n_points), np.linspace(0, 1, self.n_points)

            rho_0 = [(a,b) for a in rho_l for b in rho_r if a != b]
            v_0 = [(a,b) for a in v_l for b in v_r if a != b]


            def p(rho):
                return rho**gamma
            
            with tqdm(range(self.size), desc="Generation dataset") as pbar:
                for i in pbar:
                    self.rho[i, :(nx//2), 0] = torch.tensor(rho_0[i][0]).repeat(nx//2)
                    self.rho[i, (nx//2):, 0] = torch.tensor(rho_0[i][1]).repeat(nx//2)
                    self.v[i, :(nx//2), 0] = torch.tensor(v_0[i][0]).repeat(nx//2)
                    self.v[i, (nx//2):, 0] = torch.tensor(v_0[i][1]).repeat(nx//2)

                    rho, v = nn_arz.nn_arz.numerical_schemes.solve(self.rho[i].numpy(), self.v[i].numpy(), p, None, nx, nt, dx, dt, verbose=False)

                    self.rho[i] = torch.tensor(rho)
                    self.v[i] = torch.tensor(v)                    
                    self.y[i] = torch.multiply(self.v[i] + p(self.rho[i]), self.rho[i])

                    # u = (rho, v)
                    self.u[i, :, :, 0] = self.rho[i]
                    self.u[i, :, :, 1] = self.y[i]

                    self.flux[i, :, :, 0] = torch.multiply(self.rho[i], self.v[i])
                    self.flux[i, :, :, 1] = torch.multiply(self.y[i], self.v[i])

                    # Show
                    # nn_arz.nn_arz.utils.plot_matrix(self.v[i], xlabel='x', ylabel='t', title='Velocity Field v(x,t)', cmap='spring', extent=[-10, 10, 0, 6])
                    # nn_arz.nn_arz.utils.plot_matrix(self.rho[i], xlabel='x', ylabel='t', title='Density Field rho(x,t)', cmap='spring', extent=[-10, 10, 0, 6])

            # Create directory if it does not exist
            if not os.path.exists("datasets"):
                os.makedirs("datasets")
            
            # Save generated data to file
            torch.save({
                'rho': self.rho,
                'v': self.v,
                'y': self.y,
                'u': self.u,
                'flux': self.flux,
                'x': self.x,
                't': self.t
            }, self.path)
            print("Dataset saved to file.")

    def __len__(self):
        return self.size

    def __getitem__(self, idx): 
        return {
            'rho': self.rho[idx],
            'v': self.v[idx],
            'y': self.y[idx],
            'u': self.u[idx],
            'flux': self.flux[idx],
            'x': self.x[idx],
            't': self.t[idx]
        }
    
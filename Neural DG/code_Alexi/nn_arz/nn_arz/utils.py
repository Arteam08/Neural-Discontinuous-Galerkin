import subprocess
import torch
import random

import matplotlib.pyplot as plt
import numpy as np

import nn_arz.nn_arz.models

def plot_matrix(A, xlabel='', ylabel='', title='', cmap='spring', extent=None, vmax=None, vmin=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(A.T, cmap=cmap, aspect='auto', origin='lower', extent=extent, vmax=vmax, vmin=vmin)
    plt.colorbar(label='Value')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_matrices(matrices, titles=None, xlabel='', ylabel='', cmap='spring', extent=None, file=None, vmin=None, vmax=None, aspect='auto', size=[4,3]):
    row_matrices, col_matrices = matrices.shape[:2]
    fig, axes = plt.subplots(row_matrices, col_matrices, figsize=(size[0] * col_matrices, size[1] * row_matrices), sharey=True)

    # Compute global min and max for the color scale
    if vmin is None or vmax is None:
        vmin = matrices.min() #min(matrix.min() for matrix in matrices)
        vmax = matrices.max() #max(matrix.max() for matrix in matrices)

    # Handle only one row 
    if row_matrices == 1:
        axes = [axes]
    for row_idx, row_axes in enumerate(axes):  # Iterate over rows of axes
        if col_matrices == 1:
            row_axes = [row_axes]
        for col_idx, ax in enumerate(row_axes):  # Iterate over columns of axes
            matrix = matrices[row_idx, col_idx]  # Get the corresponding matrix
            im = ax.imshow(matrix.T, cmap=cmap, aspect=aspect, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
            if row_idx == row_matrices - 1:
                ax.set_xlabel(xlabel)
            if (titles is not None) and (row_idx==0):
                ax.set_title(titles[col_idx]) 
    
        fig.colorbar(im, ax=row_axes[-1], orientation='vertical', label='Value', pad=0.02)
    
    # axes[0].set_ylabel(ylabel)
    plt.tight_layout()
    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()


def get_mps_info():
    # Run system_profiler command to get display information
    gpu_info = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
    
    # Extract lines that mention the GPU name and core count
    lines = gpu_info.stdout.splitlines()
    gpu_name = ""
    core_count = ""
    
    for line in lines:
        if "Chipset Model" in line:
            gpu_name = line.split(": ")[1]  # Extract the GPU name
        elif "Total Number of Cores" in line:
            core_count = line.split(": ")[1]  # Extract the number of cores
    
    # Format  the result
    return f"{gpu_name} {core_count} cores"

def get_gpu_memory():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        memory_usage = [int(x) for x in result.stdout.strip().split('\n')]
        return memory_usage
    except FileNotFoundError:
        print("nvidia-smi not found. Please ensure it's installed and available in your PATH.")
        return None
    
def get_device(verbose=True):
    if torch.cuda.is_available():
        memory_usage = get_gpu_memory()
        if memory_usage is not None:
            # Filter out Nvidia DGX GPUs
            excluded_gpus = []
            for i in range(torch.cuda.device_count()):
                if "DGX" in torch.cuda.get_device_name(i):
                    excluded_gpus.append(i)

            # Filter memory usage for valid GPUs
            valid_gpus = [i for i in range(torch.cuda.device_count()) if i not in excluded_gpus]

            if valid_gpus:
                selected_gpu = min(valid_gpus, key=lambda i: memory_usage[i])  # Select GPU with least memory used
                device = torch.device(f'cuda:{selected_gpu}')

                if verbose:
                    print(f"Using GPU : {torch.cuda.get_device_name(selected_gpu)} (GPU {selected_gpu}) "
                          f"with {memory_usage[selected_gpu]:.2f} MiB used.")
            else:
                print("No suitable GPU found. Defaulting to CPU.")
                device = torch.device('cpu')

            if verbose:
                print(f"Using GPU : {torch.cuda.get_device_name(selected_gpu)} (GPU {selected_gpu}) "
                      f"with {memory_usage[selected_gpu]} MiB used.")
        else:
            print("Could not determine GPU memory usage. Defaulting to CUDA device 0.")
            device = torch.device('cuda:0')
    
        # device = torch.device('cuda')
        # if verbose:
        #     print("Using GPU :", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print(f"Using GPU : {get_mps_info()}")
    else: 
        device = torch.device('cpu')
        if verbose:
            print('Using cpu device')
    
    return device

def empty_cache():
    # Empty cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def torch_binomial_coefficient(n, k):
    if isinstance(n, int) and isinstance(k, int):
        n = torch.tensor(n, dtype=torch.float)
        k = torch.tensor(k, dtype=torch.float)
    
    fact = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    return torch.exp(fact)

def get_model(is_speed, is_DG, is_arz=False, has_x=False, device='cpu', depth=6, hidden=15, dx = 1e-2, dt = 1e-2, stencil=(0, 1), activation=torch.nn.ReLU, path=None, dropout=0., clamp=1.): 
    if is_arz:
        model = nn_arz.nn_arz.models.arzModel(
            stencil=stencil, 
            depth=depth, 
            hidden=hidden, 
            dx=dx, 
            dt=dt, 
            activation=activation
        )
    elif has_x:
        model = nn_arz.nn_arz.models.flowModel_x(
                stencil=stencil, 
                depth=depth, 
                hidden=hidden, 
                dx=dx, 
                dt=dt, 
                activation=activation,
        )
    elif is_speed:
        if is_DG:
            model = nn_arz.nn_arz.models.speedModelDG(
                depth=depth, 
                hidden=hidden, 
                dx=dx, 
                dt=dt, 
                activation=activation
            )
        else:
            model = nn_arz.nn_arz.models.speedModel(
                # (0, 1), 
                stencil=stencil, 
                depth=depth, 
                hidden=hidden, 
                dx=dx, 
                dt=dt, 
                activation=activation
            )
    else:
        if is_DG:
            model = nn_arz.nn_arz.models.flowModelDG(
                depth=depth, 
                hidden=hidden, 
                dx=dx, 
                dt=dt, 
                activation=activation
            )
        else:
            model = nn_arz.nn_arz.models.flowModel(
                stencil=stencil, 
                depth=depth, 
                hidden=hidden, 
                dx=dx, 
                dt=dt, 
                activation=activation,
                dropout=dropout,
                clamp=clamp
            )

    
    if path is not None:
        # First load in CPU because loading MPS models on CUDA creates issues otherwise.
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu"), weights_only=True))

    model.to(device)
    return model

def seed(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(0)

    torch.use_deterministic_algorithms(deterministic)

def legendre_roots_weights(n, device='cpu', dtype=torch.float):
    roots, weights = np.polynomial.legendre.leggauss(n)
    return torch.tensor(
        roots, 
        dtype=dtype, 
        device=device
    ), torch.tensor(
        weights, 
        dtype=dtype, 
        device=device
    )
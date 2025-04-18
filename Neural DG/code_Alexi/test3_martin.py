from Neural_fluxes import MLPFlux_2_value, CNNFlux
import nn_arz.nn_arz.lwr.fluxes
import torch
from params import *
from generate_data_martin import *

flow_func = nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield

model = MLPFlux_2_value('cpu')
dataset_path="Neural DG/code_Alexi/data/riemann_dataset.pt"
torch.autograd.set_detect_anomaly(True)


#define param_set to use
solver_params=params1

#files parameters
dataset_folder="Neural DG/code_Alexi/data"
plots_folder="Neural DG/code_Alexi/plots"
checkpoint_folder="Neural DG/code_Alexi/checkpoints"
save_name="MLP_riemann1.pt"



dx=solver_params["dx"]
n_cells = solver_params["n_cells"]
t_max = solver_params["t_max"]
dt=solver_params["dt"]
n_poly = solver_params["n_poly"]
points_per_cell = solver_params["points_per_cell"]

#generate dataset
generate_riemann_dataset(
    save_folder=dataset_folder,
    filename="riemann_dataset.pt",
    ic_boundaries=[0, 4],
    N_ic_range=4,
    dx=dx,
    n_cells=n_cells,
    t_max=t_max,
    dt=dt,
    device='cpu',
    batch_size=64,
)


model.train_to_func(
    flow_func=flow_func,
    lr=1e-3,
    n_epochs=10**4,
    batch_size=10**5,
    u_amplitude=10,
)




model.train_on_data(
    dataset_path=dataset_path,
    solver_params=solver_params,
    n_epochs=10**4,
    batch_size=10**2,
    lr=1e-3,
)

model.save_model(
    save_folder=checkpoint_folder,
    filename=save_name,
)




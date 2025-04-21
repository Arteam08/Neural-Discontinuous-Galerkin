from Neural_fluxes import MLPFlux_2_value, CNNFlux
import nn_arz.nn_arz.lwr.fluxes
import torch
from Solver_Params import *
from generate_data_martin import *

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

#define param_set to use
solver_params=solver_params1
# solver_params['device']=device


#model and flux definition
hidden_dims=[64,64]
flow_func = nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield


dataset_path="Neural DG/code_Alexi/data/riemann_dataset.pt"
torch.autograd.set_detect_anomaly(True)

#parameters of the run 
GENERATE_DATA=False #load data if available
PRE_TRAIN=False# use the train to func
DATA_TRAIN=True#actually train on the data aftewards



#files parameters
dataset_folder="Neural DG/code_Alexi/data"
plots_folder="Neural DG/code_Alexi/plots"
checkpoint_folder="Neural DG/code_Alexi/checkpoints"
data_train_name="MLP_riemann1.pt"
data_train_path=os.path.join(checkpoint_folder,data_train_name)

pre_train_name="pretrain.pt"
pre_train_path=os.path.join(checkpoint_folder,pre_train_name)

#copy the file parameters
dx=solver_params["dx"]
n_cells = solver_params["n_cells"]
t_max = solver_params["t_max"]
dt=solver_params["dt"]
n_poly = solver_params["n_poly"]
points_per_cell = solver_params["points_per_cell"]

#generate dataset

if GENERATE_DATA:
    generate_riemann_dataset(
        save_folder=dataset_folder,
        filename="riemann_dataset.pt",
        ic_boundaries=[0, 4],
        N_ic_range=4,
        dx=dx,
        n_cells=n_cells,
        t_max=t_max,
        dt=dt,
        device=device,
        batch_size=64,
    )

    

if PRE_TRAIN:
    model = MLPFlux_2_value(device=device, hidden_dims=hidden_dims)
    model.to(device)
    model.train_to_func(
        flow_func=flow_func,
        lr=1e-3,
        n_epochs=1*10**4,
        batch_size=10**5,
        u_amplitude=5,
    )

    model.save(checkpoint_folder,pre_train_name)
else:
    model=MLPFlux_2_value.load(filepath=pre_train_path, device=device)



if DATA_TRAIN:
    model.train_on_data(
            dataset_path=dataset_path,
            solver_params=solver_params,
            n_epochs=10**4,
            batch_size=10**2,
            lr=1e-4,
    )





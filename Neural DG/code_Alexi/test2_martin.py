from generate_data_martin import *
from eval_and_plot_funcs_martin import *
from solvers_martin import DG_solver
import nn_arz.nn_arz.lwr.fluxes

####parameters
dx= 1e-2
n_cells = 100
t_max = 600
dt=1e-4
n_poly = 2
points_per_cell = 20
flux_function = nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield


#####saving
dataset_folder="Neural DG/code_Alexi/data"
plots_folder="Neural DG/code_Alexi/plots"

riemann_dataset_path = os.path.join(dataset_folder, "riemann_dataset.pt")
piecewise_dataset_path= os.path.join(dataset_folder, "piecewise_constant_dataset.pt")

# generate_riemann_dataset(
#     save_folder=dataset_folder,
#     filename="riemann_dataset.pt",
#     ic_boundaries=[0, 4],
#     N_ic_range=4,
#     dx=dx,
#     n_cells=n_cells,
#     t_max=t_max,
#     dt=dt,
#     device='cpu',
#     batch_size=64,
# )

# generate_piecewise_constant_dataset(
#     save_folder=dataset_folder,
#     filename="piecewise_constant_dataset.pt",
#     ic_boundaries=[0, 4],
#     N_ic_range=4,
#     dx=dx,
#     n_cells=n_cells,
#     t_max=t_max,
#     dt=dt,
#     device='cpu',
#     batch_size=64,
# )

solver= DG_solver(
    t_max=t_max,
    dx=dx,
    dt=dt,
    n_cells=n_cells,
    flow_func=flux_function,
    points_per_cell=points_per_cell,
    n_poly=n_poly,
    device='cpu',
)

run_solver(
    solver=solver,
    dataset_path=riemann_dataset_path,
    batch_size=64,
    plot_path=os.path.join(plots_folder, "heatmaps"),
)
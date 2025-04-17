from generate_data_martin import *
from eval_and_plot_funcs_martin import *
from solvers_martin import DG_solver
import nn_arz.nn_arz.lwr.fluxes

####parameters
dx= 0.5*1e-2
n_cells = 300
t_max = 1000
dt=1e-3
n_poly = 2
points_per_cell = 20
flux_function = nn_arz.nn_arz.lwr.fluxes.godunov_flux_greenshield

IS_RIEMANN = False

#####saving paths
dataset_folder="Neural DG/code_Alexi/data"
plots_folder="Neural DG/code_Alexi/plots"

riemann_dataset_path = os.path.join(dataset_folder, "riemann_dataset.pt")
piecewise_dataset_path= os.path.join(dataset_folder, "piecewise_constant_dataset.pt")

dataset_path=riemann_dataset_path if IS_RIEMANN else piecewise_dataset_path


if IS_RIEMANN:

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

else:

    generate_piecewise_constant_dataset(
    save_folder=dataset_folder,
    filename="piecewise_constant_dataset.pt",
    N_pieces=4,
    amplitude= 4.,
    n_ic=20,
    dx = dx,
    n_cells = n_cells,
    points_per_cell= 10,
    t_max= t_max,
    dt = dt,
    batch_size= 3,  # (batch,)
    ratio = 10,
    device = "cpu",
)


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
    dataset_path=dataset_path,
    batch_size=64,
    plot_path=os.path.join(plots_folder, "heatmaps"),
)
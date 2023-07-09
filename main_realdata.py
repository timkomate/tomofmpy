from utils.tomo_eikonal import Eikonal_Inversion
import numpy as np
import scipy.optimize


if __name__ == "__main__":
    y = 1000
    x = 1400
    ny = 20
    nx = 28
    # ny = 3
    # nx = 4

    inv = Eikonal_Inversion(
        (y, x),
        (ny, nx),
        "./input_data_80s_xy.csv",
        root_folder="./real_data_80s_DA2",
        eta=2,
        epsilon=2,
        BL=(11, 44.4),
        transform=True,
    )

    # Linearized inversion
    x0 = np.full((ny * nx), 3.87)
    inv.add_starting_model(x0)
    # inv.run_linearization(method="L-BFGS-B", jac = "2-points", options={"maxiter": 50, "disp": True}, bounds=scipy.optimize.Bounds(3.6,4.2))

    # Nonlinear inversion
    bounds = [(3.6, 4.2)] * ny * nx
    # print(inv.run_nonlinear(mode = "GA", bounds = bounds, maxiter = 100, popsize = 10, disp = True, workers = 1, polish = True))
    print(
        inv.run_nonlinear(mode="DA", bounds=bounds, no_local_search=False, maxfun=5000)
    )
    # options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    # optimizer_options = {"iter": 1000}
    # max_bound = 2.5 * np.ones(ny * nx)
    # min_bound = 1.5 * np.ones(ny * nx)
    # bounds = (min_bound, max_bound)
    # print(inv.run_nonlinear(mode = "PSO",n_particles=10, options=options, bounds=bounds))

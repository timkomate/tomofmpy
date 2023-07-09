from utils.tomo_eikonal import Eikonal_Inversion
import numpy as np
import scipy.optimize


if __name__ == "__main__":
    y = 120
    x = 120
    ny = 30
    nx = 30

    y = 120
    x = 120
    ny = 24
    nx = 24

    inv = Eikonal_Inversion(
        (y, x),
        (ny, nx),
        "./synthetic_measurements_xy_noise.csv",
        root_folder="./test_syn_cb_noise",
        eta=0.0,
        epsilon=0.0,
    )

    # Linearized inversion
    x0 = np.full((ny * nx), 2.0)
    inv.add_starting_model(x0)
    inv.run_linearization(
        method="L-BFGS-B",
        jac="2-points",
        options={"maxiter": 50, "disp": True},
        bounds=scipy.optimize.Bounds(1.5, 3),
    )

    # Nonlinear inversion
    # bounds = [(1.8, 2.4)] * ny*nx
    # print(inv.run_nonlinear(mode = "GA", bounds = bounds, maxiter = 100, popsize = 10, disp = True, workers = 1, polish = True))
    print(
        inv.run_nonlinear(mode="DA", bounds=bounds, no_local_search=False, maxfun=2000)
    )
    # options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    # optimizer_options = {"iter": 1000}
    # max_bound = 2.5 * np.ones(ny * nx)
    # min_bound = 1.5 * np.ones(ny * nx)
    # bounds = (min_bound, max_bound)
    # print(inv.run_nonlinear(mode = "PSO",n_particles=10, options=options, bounds=bounds))

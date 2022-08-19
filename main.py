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
    ny = 12
    nx = 12

    inv = Eikonal_Inversion((y,x), (ny,nx), "./synthetic_measurements_xy.csv", root_folder="./test_syn", eta = 0.0, epsilon=0.0)

    #Linearized inversion
    #x0 = np.full((ny*nx), 2.05)
    #inv.add_starting_model(x0)
    #inv.run_linearization(method="L-BFGS-B", jac = "2-points", options={"maxiter": 50, "disp": True}, bounds=scipy.optimize.Bounds(2,4))

    #Nonlinear inversion
    bounds = [(1.8, 2.4)] * ny*nx
    #print(inv.run_nonlinear(bounds = bounds, maxiter = 10, popsize = 10, disp = True, workers = 1, polish = False))
    print(inv.run_nonlinear(mode = "DA", bounds = bounds, no_local_search=True))
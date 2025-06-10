import os
import numpy as np
import scipy.optimize
from scipy.spatial import ConvexHull
import logging

from tomoFMpy.core.solver import Eikonal_Solver
from tomoFMpy.utils.plotting import plot_model_grid

logger = logging.getLogger(__name__)


class Eikonal_Inversion:
    """
    Core inversion driver for Eikonal problems. Handles misfit, regularization,
    and optimization loops. Uses EikonalSolver as backend; no direct I/O except
    saving outputs.
    """

    def __init__(
        self,
        modelspace,
        gridsize,
        measurement_csv,
        output_folder="./",
        bl_corner=(0, 0),
        epsilon=0,
        eta=0,
        use_latlon=False,
    ):
        self.y, self.x = modelspace
        self.ny, self.nx = gridsize
        self.use_latlon = use_latlon

        self.n = self.nx * self.ny
        self.dy, self.dx = self.y / gridsize[0], self.x / gridsize[1]
        self.yaxis = np.arange(self.ny) * self.dy
        self.xaxis = np.arange(self.nx) * self.dx
        self.measurement_csv = measurement_csv
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        logger.debug("Output folder: %s", self.output_folder)
        self.epsilon = epsilon
        self.eta = eta
        self.residuals = []

        self.solver = Eikonal_Solver(
            grid=np.zeros((self.ny, self.nx)),
            gridsize=(self.dy, self.dx),
            measurements_csv=self.measurement_csv,
            bl_corner=bl_corner,
        )
        if self.use_latlon:
            self.solver.transform_to_xy()
        self._save_station_files()

    def add_starting_model(self, x0):
        """
        Store the initial model vector (flattened) for perturbation.
        """
        self.x0 = x0

    def _build_laplacian(self):
        """
        Build sparse Laplacian matrix in dense form for roughness penalty.
        """
        n = self.ny * self.nx
        D = np.zeros((n, n), dtype=float)
        idx = 0
        for i in range(self.ny):
            for j in range(self.nx):
                nbr = 0
                row = np.zeros((self.ny, self.nx), dtype=float)
                if i > 0:
                    nbr += 1
                    row[i - 1, j] = -1
                if i < self.ny - 1:
                    nbr += 1
                    row[i + 1, j] = -1
                if j > 0:
                    nbr += 1
                    row[i, j - 1] = -1
                if j < self.nx - 1:
                    nbr += 1
                    row[i, j + 1] = -1
                row[i, j] = nbr
                D[idx, :] = row.reshape(n)
                idx += 1
        return D

    def _perturbation(self, xi):
        """L2 perturbation = ||xi - x0||^2"""
        return float(((xi - self.x0) ** 2).sum())

    def _roughness(self, vec):
        """Roughness = vec^T D vec"""
        D = self._build_laplacian()
        return float(vec @ (D @ vec))

    def misfit(self, model_vec):
        """
        Compute misfit = sqrt{ data_misfit + eps*perturbation + eta*roughness }
        """
        grid = model_vec.reshape(self.ny, self.nx)
        self.solver.update_model(grid)
        self.solver.solve()
        self.solver.calculate_traveltimes()
        data_cost = self.solver.calc_residuals()
        reg = 0.0
        if self.epsilon:
            reg += self.epsilon * self._perturbation(model_vec)
        if self.eta:
            reg += self.eta * self._roughness(model_vec)
        return np.sqrt(data_cost + reg)

    def _on_iteration(self, vec, cost, iteration):
        """
        Common callback: log, append residual, save model, xyz, figure.
        """
        logger.info("Iteration %d: residual=%.4f", iteration, cost)
        self.residuals.append(cost)
        self._save_model_csv(vec, iteration)
        self._save_model_xyz(vec, iteration)
        self._save_model_png(vec, iteration)

    def _save_model_csv(self, vec, iteration):
        sol = vec.reshape(self.ny, self.nx)
        path = os.path.join(self.output_folder, f"{iteration}.csv")
        np.savetxt(path, sol, fmt="%.2f", delimiter=",")

    def _save_model_xyz(self, vec, iteration):
        outp = os.path.join(self.output_folder, f"{iteration}.xyz")
        flat = vec.flatten()
        idx = 0
        with open(outp, "w") as f:
            for y in self.yaxis:
                for x in self.xaxis:
                    if self.use_latlon:
                        lon, lat = self.solver.inv_transformer.transform(
                            x * 1000 + self.solver.base_xy[0],
                            y * 1000 + self.solver.base_xy[1],
                        )
                        f.write(f"{lon:.3f} {lat:.3f} {flat[idx]:.3f}\n")
                    else:
                        f.write(f"{x:.3f} {y:.3f} {flat[idx]:.3f}\n")
                    idx += 1

    def _save_model_png(self, vec, iteration):
        sol = vec.reshape(self.ny, self.nx)
        png = os.path.join(self.output_folder, f"{iteration}.png")
        plot_model_grid(self.xaxis, self.yaxis, sol, output_path=png)

    def save_residuals(self):
        arr = np.array(self.residuals, dtype=float)
        path = os.path.join(self.output_folder, "residuals.txt")
        np.savetxt(path, arr, fmt="%.4f")

    def _save_station_files(self):
        stations = self.solver.get_unique_stations(self.use_latlon)
        hull = ConvexHull(stations)
        pts = stations[hull.vertices]
        np.savetxt(
            os.path.join(self.output_folder, "stations.xy"), stations, fmt="%.2f"
        )
        np.savetxt(os.path.join(self.output_folder, "hull.xy"), pts, fmt="%.2f")
        logger.info(
            "Wrote stations (n=%d) and hull to %s", len(stations), self.output_folder
        )

    def run_linear(self, *args, **kwargs):
        if not hasattr(self, "x0"):
            raise RuntimeError("Call add_starting_model() first.")
        self.Nfeval = 0
        self._on_iteration(self.x0, self.misfit(self.x0), self.Nfeval)

        def cb(xk):
            cost = self.misfit(xk)
            self._on_iteration(xk, cost, self.Nfeval)
            self.Nfeval += 1

        result = scipy.optimize.minimize(
            fun=lambda v: self.misfit(v), x0=self.x0, callback=cb, *args, **kwargs
        )
        # final
        self._on_iteration(result.x, self.misfit(result.x), self.Nfeval)
        self.save_residuals()
        return result

    def fitness_func_PSO(self, x: np.array, dim) -> np.float64:
        x = x.reshape(dim, self.ny * self.nx)
        fittnesses = np.apply_along_axis(self.fitness_func, axis=1, arr=x)
        return fittnesses

    def callbackF_EA(self, Xi, convergence):
        # print(convergence)
        res = self.fitness_func(Xi)
        iteration = self.Nfeval
        print(f"iteration: {iteration}, residual {res:.3f}")
        self.residuals.append(res)
        self._save_model_csv(Xi, iteration)
        self._save_model_xyz(Xi, iteration)
        self._save_model_png(Xi, iteration)
        self.Nfeval += 1

    def callbackF_DA(self, Xi, f, contex):
        iteration = self.Nfeval
        print(f"iteration: {self.Nfeval}, residual {f:.3f}")
        self._save_model_csv(Xi, iteration)
        self._save_model_xyz(Xi, iteration)
        self._save_model_png(Xi, iteration)
        self.Nfeval += 1

    def callbackF_PSO(self, Xi):
        iteration = self.Nfeval
        res = self.fitness_func(Xi)
        print(f"iteration: {self.Nfeval}, residual {res:.3f}")
        self.residuals.append(res)
        self._save_model_csv(Xi, iteration)
        self._save_model_xyz(Xi, iteration)
        self._save_model_png(Xi, iteration)
        self.Nfeval += 1

    def run_nonlinear(self, mode, *args, **kwargs):
        self.Nfeval = 0
        if mode == "GA":
            solver = scipy.optimize.differential_evolution(
                func=self.fitness_func, *args, **kwargs, callback=self.callbackF_EA
            )
        elif mode == "DA":
            solver = scipy.optimize.dual_annealing(
                func=self.fitness_func, *args, **kwargs, callback=self.callbackF_DA
            )
        elif mode == "PSO":
            import pyswarms as ps

            optimizer = ps.single.GlobalBestPSO(
                dimensions=self.ny * self.nx, *args, **kwargs
            )
            solver = optimizer.optimize(self.fitness_func_PSO, iters=5, dim=10)
        else:
            raise NotImplementedError
        return solver

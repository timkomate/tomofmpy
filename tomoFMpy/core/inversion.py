import os
import numpy as np
import scipy.optimize
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from tomoFMpy.core.solver import EikonalSolver

class Eikonal_Inversion:
    def __init__(
        self,
        modelspace,
        gridsize,
        filename,
        root_folder="./",
        BL=(0, 0),
        epsilon=0,
        eta=0,
        transform=False,
    ):
        self.y, self.x = modelspace
        self.ny, self.nx = gridsize
        self.transform = transform

        self.n = self.nx * self.ny
        self.dy, self.dx = self.y / gridsize[0], self.x / gridsize[1]
        self.yaxis = np.arange(self.ny) * self.dy
        self.xaxis = np.arange(self.nx) * self.dx
        self.filename = filename
        self.root_folder = root_folder
        if not os.path.exists(self.root_folder):
            os.makedirs(root_folder)
        self.epsilon = epsilon
        self.eta = eta

        self.residuals = []
        self.eik = Eikonal_Solver(
            np.zeros((self.ny, self.nx)),
            gridsize=(self.dy, self.dx),
            filename=self.filename,
            BL=BL,
        )
        if self.transform:
            self.eik.transform2xy()
        self.save_stations()

    def fitness_func(self, x: np.array) -> np.float64:
        velocity_model = x.reshape(self.ny, self.nx)
        self.eik.update_model(velocity_model)
        self.eik.solve()
        self.eik.calc_traveltimes()
        fitness = self.eik.calc_residuals()
        if self.epsilon:
            fitness += self.epsilon * self.calc_perturbation(x)
        if self.eta:
            fitness += self.eta * self.calc_roughness(x)
        return np.sqrt(fitness)

    def fitness_func_PSO(self, x: np.array, dim) -> np.float64:
        x = x.reshape(dim, self.ny * self.nx)
        fittnesses = np.apply_along_axis(self.fitness_func, axis=1, arr=x)
        return fittnesses

    def fitness_func_PSO(self, x: np.array, dim) -> np.float64:
        x = x.reshape(dim, self.ny * self.nx)
        fittnesses = np.apply_along_axis(self.fitness_func, axis=1, arr=x)
        return fittnesses

    def calc_perturbation(self, x):
        return (x - self.x0).T @ (x - self.x0)

    def calc_roughness(self, x):
        D = self.calc_D()
        return x.T @ D @ x

    def calc_D(self):
        D = np.zeros((self.n, self.n), dtype=np.float64)
        c = 0
        for i in np.arange(self.ny):
            for j in np.arange(self.nx):
                D_tmp = np.zeros((self.ny, self.nx), dtype=np.float64)
                cc = 0
                if i - 1 >= 0:
                    cc += 1
                    D_tmp[i - 1, j] = -1
                if i + 1 < self.ny:
                    cc += 1
                    D_tmp[i + 1, j] = -1
                if j - 1 >= 0:
                    cc += 1
                    D_tmp[i, j - 1] = -1
                if j + 1 < self.nx:
                    cc += 1
                    D_tmp[i, j + 1] = -1
                D_tmp[i, j] = cc
                D[c, :] = D_tmp.reshape(self.n)
                c += 1
        return D

    def callbackF(self, Xi):
        res = self.fitness_func(Xi)
        print(f"iteration: {self.Nfeval}, residual {res:.3f}")
        self.residuals.append(res)
        self.save_model(Xi, self.Nfeval)
        self.save_model2(Xi, self.Nfeval)
        self.save_figure(Xi, self.Nfeval)
        self.Nfeval += 1

    def callbackF_EA(self, Xi, convergence):
        # print(convergence)
        res = self.fitness_func(Xi)
        print(f"iteration: {self.Nfeval}, residual {res:.3f}")
        self.residuals.append(res)
        self.save_model(Xi, self.Nfeval)
        self.save_model2(Xi, self.Nfeval)
        self.save_figure(Xi, self.Nfeval)
        self.Nfeval += 1

    def callbackF_DA(self, Xi, f, contex):
        print(f"iteration: {self.Nfeval}, residual {f:.3f}")
        self.residuals.append(f)
        self.save_model(Xi, self.Nfeval)
        self.save_model2(Xi, self.Nfeval)
        self.save_figure(Xi, self.Nfeval)
        self.Nfeval += 1

    def callbackF_PSO(self, Xi):
        res = self.fitness_func(Xi)
        print(f"iteration: {self.Nfeval}, residual {res:.3f}")
        self.residuals.append(f)
        self.save_model(Xi, self.Nfeval)
        self.save_model2(Xi, self.Nfeval)
        self.save_figure(Xi, self.Nfeval)
        self.Nfeval += 1

    def callbackF_PSO(self, Xi):
        res = self.fitness_func(Xi)
        print(f"iteration: {self.Nfeval}, residual {res:.3f}")
        self.residuals.append(f)
        self.save_model(Xi, self.Nfeval)
        self.save_model2(Xi, self.Nfeval)
        self.save_figure(Xi, self.Nfeval)
        self.Nfeval += 1

    def save_model(self, Xi, iteration):
        sol = Xi.reshape(self.ny, self.nx)
        np.savetxt(
            f"{self.root_folder}/{iteration}.csv", sol, fmt="%.2f", delimiter=","
        )

    def save_model2(self, Xi, iteration):
        output = open(f"{self.root_folder}/{iteration}.xyz", "w")
        c = 0
        for i in self.yaxis:
            for j in self.xaxis:
                if self.transform:
                    x, y = self.eik.transform2latlon(j, i)
                else:
                    x, y = j, i
                output.write(f"{x:.3f} {y:.3f} {Xi[c]:.3f}\n")
                c += 1
        output.close()

    def save_figure(self, Xi, iteration):
        sol = Xi.reshape(self.ny, self.nx)
        plt.figure()
        plt.pcolor(self.xaxis, self.yaxis, sol)  # vmax=1.9, vmin=2.1)
        plt.colorbar()
        plt.savefig(f"{self.root_folder}/{iteration}.png")
        plt.clf()
        plt.close()

    def save_residuals(self):
        res = np.array(self.residuals)
        np.savetxt(f"{self.root_folder}/residuals.txt", X=res.T, fmt="%.2f")

    def save_stations(self):
        stations = self.eik.get_unique_stations(self.transform)
        hull = ConvexHull(stations)
        wrapper = stations[hull.vertices]
        np.savetxt("{}/hull.xy".format(self.root_folder), wrapper, fmt="%.2f")
        np.savetxt("{}/stations.xy".format(self.root_folder), stations, fmt="%.2f")

    def add_starting_model(self, x0):
        self.x0 = x0

    def run_linearization(self, *args, **kwargs):
        self.Nfeval = 0
        self.callbackF(self.x0)
        solver = scipy.optimize.minimize(
            self.fitness_func, self.x0, *args, **kwargs, callback=self.callbackF
        )
        self.callbackF(solver.x)
        self.save_residuals()
        return solver

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


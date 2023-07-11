from fteikpy import Eikonal2D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import scipy.optimize
from scipy.spatial import ConvexHull
import os


class Eikonal_Solver(Eikonal2D):
    def __init__(self, grid, gridsize, filename, origin=None, BL=(0, 0)):
        super().__init__(grid, gridsize, origin)
        self.df = self._add_measurements(filename)
        self.sigma = 1 / self.df["sigma"].to_numpy() if "sigma" in self.df.columns else np.ones(len(self.df))
        self.Cd = np.diag(self.sigma)
        self.traveltimes = np.zeros(len(self.df))
        self.base = BL
        self.transformer = None
        self.inv_transformer = None

    def _init_transformer(self):
        lon_min, lat_min = self.df[["lons", "lats", "lonr", "latr"]].min().values
        lon_max, lat_max = self.df[["lons", "lats", "lonr", "latr"]].max().values
        lon = lon_min + (lon_max - lon_min) / 2
        lat = lat_min + (lat_max - lat_min) / 2
        crs = pyproj.CRS.from_string(f"+proj=laea +lat_0={lat} +lon_0={lon} +lat_b=0 +ellps=WGS84")
        self.transformer = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
        self.inv_transformer = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
        self.base = self.transformer.transform(*self.base)

    def transform2xy(self):
        if self.transformer is None:
            self._init_transformer()
        columns = ["lons", "lats", "lonr", "latr"]
        xy_columns = ["xs", "ys", "xr", "yr"]
        for column, xy_column in zip(columns, xy_columns):
            coordinates = self.df[column]
            x, y = self.transformer.transform(*coordinates)
            self.df[xy_column] = (x - self.base[0]) / 1000 if xy_column.endswith("s") else (x - self.base[0]) / 1000

    def transform2latlon(self, X, Y):
        if self.inv_transformer is None:
            self._init_transformer()
        x, y = self.inv_transformer.transform(X * 1000 + self.base[0], Y * 1000 + self.base[1])
        return x, y

    def _add_measurements(self, filename):
        df = pd.read_csv(filename, dtype={"source_id": np.uint32})
        return df

    def _get_sources(self):
        sources = self.df["source_id"].unique()
        s = self.df.loc[self.df.groupby("source_id").head(1).index][["ys", "xs"]].values
        return s

    def get_unique_stations(self, transform):
        stations = self.df[["xr", "yr"]] if transform else self.df[["lonr", "latr"]]
        return stations.values

    def _get_receivers(self, i):
        subset = self.df[self.df["source_id"] == i]
        r = subset[["yr", "xr"]].values
        return r

    def solve(self, return_gradient=False, nsweep=2):
        sources = self._get_sources()
        self.tt = super().solve(sources=sources, nsweep=nsweep, return_gradient=return_gradient)
        return self.tt

    def calc_traveltimes(self):
        offset = 0
        for i, grid in enumerate(self.tt):
            receivers = self._get_receivers(i)
            n = receivers.shape[0]
            self.traveltimes[offset:offset+n] = grid(receivers)
            offset += n

    def get_residuals(self):
        return self.df["tt"].to_numpy() - self.traveltimes

    def calc_residuals(self):
        residuals = self.get_residuals()
        return (residuals @ self.Cd @ residuals) / len(self.df)

    def save_measurements(self, filename):
        self.df["tt"] = self.traveltimes
        self.df.to_csv(filename, index=False)

    def add_noise(self, sigma, mu=0):
        noise = np.random.normal(mu, sigma, size=self.traveltimes.size)
        self.traveltimes += noise


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
        os.makedirs(root_folder, exist_ok=True)
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

    def calc_perturbation(self, x):
        return (x - self.x0).T @ (x - self.x0)

    def calc_roughness(self, x):
        D = self.calc_D()
        return x.T @ D @ x

    def calc_D(self):
        D = np.zeros((self.n, self.n), dtype=np.float64)
        c = 0
        for i in range(self.ny):
            for j in range(self.nx):
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
        self.residuals.append(res)
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

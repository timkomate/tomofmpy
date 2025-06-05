from fteikpy import Eikonal2D
import pandas as pd
import numpy as np
import pyproj


class Eikonal_Solver(Eikonal2D):
    def __init__(self, grid, gridsize, filename, origin=None, BL=(0, 0)):
        super().__init__(grid, gridsize, origin)
        self.df = self._add_measurements(filename)
        if "sigma" in self.df.columns:
            sigma = 1 / self.df["sigma"].to_numpy()
        else:
            sigma = np.ones((len(self.df)))
        self.Cd = np.zeros((sigma.size, sigma.size))
        np.fill_diagonal(self.Cd, sigma)
        self.traveltimes = np.zeros(len(self.df))
        self.base = BL

    def _init_transformer(self):
        lon_min = np.min([np.min(self.df["lons"]), np.min(self.df["lonr"])])
        lat_min = np.min([np.min(self.df["lats"]), np.min(self.df["latr"])])
        lon_max = np.max([np.max(self.df["lons"]), np.max(self.df["lonr"])])
        lat_max = np.max([np.max(self.df["lats"]), np.max(self.df["latr"])])
        lon = lon_min + (lon_max - lon_min) / 2
        lat = lat_min + (lat_max - lat_min) / 2
        crs = pyproj.CRS.from_proj4(
            f"+proj=laea +lat_0={lat} +lon_0={lon} +lat_b=0 +ellps=WGS84"
        )
        self.transformer = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
        self.inv_transformer = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)

    def transform2xy(self):
        self._init_transformer()
        self.base = self.transformer.transform(self.base[0], self.base[1])
        # self.TR = self.transformer.transform(self.TR[0],self.TR[1])
        Es, Ns = self.transformer.transform(self.df["lons"], self.df["lats"])
        Er, Nr = self.transformer.transform(self.df["lonr"], self.df["latr"])
        self.df["xs"] = (Es - self.base[0]) / 1000
        self.df["ys"] = (Ns - self.base[1]) / 1000
        self.df["xr"] = (Er - self.base[0]) / 1000
        self.df["yr"] = (Nr - self.base[1]) / 1000

    def transform2latlon(self, X, Y):
        # self._init_transformer()
        lons, lats = self.inv_transformer.transform(
            (X * 1000 + self.base[0]), (Y * 1000 + self.base[1])
        )
        return lons, lats

    def update_model(self, x):
        self._grid = x

    def _add_measurements(self, filename):
        df = pd.read_csv(filepath_or_buffer=filename, dtype={"source_id": np.uint32})
        return df

    def _get_sources(self):
        sources = pd.unique(self.df["source_id"])
        s = np.zeros((sources.size, 2))
        for i, source in enumerate(sources):
            subset = self.df[self.df["source_id"] == source]
            s[i] = subset[["ys", "xs"]].iloc[0].values
        return s

    def get_unique_stations(self, transform):
        if not transform:
            stations = self.df[["xr", "yr"]]
        else:
            stations = self.df[["lonr", "latr"]]
        return stations.to_numpy()

    def _get_receivers(self, i):
        subset = self.df[self.df["source_id"] == i]
        r = subset[["yr", "xr"]].values
        return r

    def solve(self, return_gradient=False, nsweep=2):
        sources = self._get_sources()
        self.tt = Eikonal2D.solve(
            self, sources=sources, nsweep=nsweep, return_gradient=return_gradient
        )
        return self.tt

    def calc_traveltimes(self):
        offset = 0
        for i, grid in enumerate(self.tt):
            points = self._get_receivers(i)
            n = points.shape[0]
            self.traveltimes[offset : n + offset] = grid(points)
            offset += n

    def get_residuals(self):
        return self.df["tt"].to_numpy() - self.traveltimes

    def calc_residuals(self):
        residuals = self.get_residuals()
        return (residuals @ self.Cd @ residuals) / len(self.df)

    def save_measurements(self, filename):
        self.df["tt"] = self.traveltimes
        self.df.to_csv(path_or_buf="{}".format(filename), index=False)

    def add_noise(self, sigma, mu=0):
        noise = np.random.normal(mu, sigma, size=(self.traveltimes.size))
        self.traveltimes += noise

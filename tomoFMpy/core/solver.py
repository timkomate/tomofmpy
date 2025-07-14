import logging
import os

import numpy as np
import pandas as pd
import pyproj
from fteikpy import Eikonal2D

from tomoFMpy.utils import io

logger = logging.getLogger(__name__)


class Eikonal_Solver(Eikonal2D):
    """
    Wrapper around fteikpy.Eikonal2D that reads measurement CSV,
    applies optional lat/lon transform, solves for travel times,
    and manages residuals and noise.
    """

    def __init__(self, grid, gridsize, measurements_csv, origin=None, bl_corner=(0, 0)):
        """
        Wrapper around fteikpy.Eikonal2D that reads measurement CSV,
        applies optional lat/lon transform, solves for travel times,
        and manages residuals and noise.
        """
        # Initialize base class
        super().__init__(grid, gridsize, origin)
        logger.debug(
            "Initializing EikonalSolver with grid shape %s, gridsize %s",
            grid.shape,
            gridsize,
        )
        # Load measurements
        self.df = io.load_measurements_csv(measurements_csv)
        logger.info(
            "Loaded %d measurement rows from %s", len(self.df), measurements_csv
        )
        self.Cd = self._build_covariance()
        logger.debug("Constructed covariance matrix of size %s", self.Cd.shape)
        self.traveltimes = np.zeros(len(self.df), dtype=float)
        self.base = bl_corner
        logger.debug("Base corner set to %s", self.base)
        self.transformer = None
        self.inv_transformer = None

    def _initialize_transformers(self):
        """
        Create pyproj transformers for converting between lat/lon and local East/North (LAEA),
        centered at the midpoint of all station coordinates (min/max of lons/lats).
        """
        logger.debug("Initializing latitude/longitude transformers")
        lons = np.concatenate([self.df["lons"], self.df["lonr"]])
        lats = np.concatenate([self.df["lats"], self.df["latr"]])
        lon = lons.min() + (lons.max() - lons.min()) / 2
        lat = lats.min() + (lats.max() - lats.min()) / 2
        crs = pyproj.CRS.from_proj4(
            f"+proj=laea +lat_0={lat} +lon_0={lon} +lat_b=0 +ellps=WGS84"
        )
        self.transformer = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
        self.inv_transformer = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
        self.base_xy = self.transformer.transform(self.base[0], self.base[1])
        logger.info(
            "Transformers initialized for LAEA centered at (%.4f, %.4f)", lon, lat
        )

    def transform_to_xy(self):
        """
        Convert lat/lon in self.df to local (x, y) in kilometers, relative to base_corner.
        After this call, self.base_corner is updated to (East, North) in meters,
        and df gains columns: xs, ys, xr, yr (in km).
        Requires self.df to have columns: lons, lats, lonr, latr.
        """
        logger.info("Transforming lat/lon to local XY (km)")
        if self.transformer is None or self.inv_transformer is None:
            self._initialize_transformers()
        Es, Ns = self.transformer.transform(self.df["lons"], self.df["lats"])
        Er, Nr = self.transformer.transform(self.df["lonr"], self.df["latr"])
        self.df["xs"] = (Es - self.base_xy[0]) / 1000
        self.df["ys"] = (Ns - self.base_xy[1]) / 1000
        self.df["xr"] = (Er - self.base_xy[0]) / 1000
        self.df["yr"] = (Nr - self.base_xy[1]) / 1000

    def transform_to_latlon(self):
        """
        Convert local (x, y) in self.df to lat/lon in, relative to base_corner.
        After this call, self.base_corner is updated,
        and df gains columns: lons, lats, lonr, latr.
        Requires self.df to have columns: xr, yr, xs, ys.
        """
        logger.info("Transforming local XY (km) to lat/lon")
        if self.transformer is None or self.inv_transformer is None:
            self._initialize_transformers()

        self.df["lons"], self.df["lats"] = self.inv_transformer.transform(
            (self.df["xs"] * 1000 + self.base_xy[0]),
            (self.df["ys"] * 1000 + self.base_xy[1]),
        )

        self.df["lonr"], self.df["latr"] = self.inv_transformer.transform(
            (self.df["xr"] * 1000 + self.base_xy[0]),
            (self.df["yr"] * 1000 + self.base_xy[1]),
        )

    def update_model(self, x):
        """
        Replace the internal velocity grid with new_grid.

        Args:
            new_grid: 2D array matching original grid shape.
        """
        logger.debug("Updating velocity model.")
        self._grid = x

    def _build_covariance(self):
        """
        Construct covariance matrix Cd from sigma column (1/sigma on diagonal).
        If sigma missing, returns identity.
        """
        if "sigma" in self.df.columns:
            sigma_vec = 1.0 / self.df["sigma"].to_numpy(dtype=float)
        else:
            sigma_vec = np.ones(len(self.df), dtype=float)
        Cd = np.zeros((sigma_vec.size, sigma_vec.size), dtype=float)
        np.fill_diagonal(Cd, sigma_vec)
        return Cd

    def _get_sources(self):
        """
        Extract unique source locations in (ys, xs) order.
        Returns:
            sources: array of shape (num_sources, 2).
        """
        unique_ids = pd.unique(self.df["source_id"])
        sources = np.zeros((len(unique_ids), 2), dtype=float)
        for idx, src_id in enumerate(unique_ids):
            row = self.df[self.df["source_id"] == src_id].iloc[0]
            sources[idx, :] = [row["ys"], row["xs"]]
        return sources

    def get_unique_stations(self, transform):
        """
        Return an array of unique receiver station coordinates.
        If transform=False, returns (xr, yr); else returns (lonr, latr).

        Args:
            transform: Bool flag. If False, return local XY; otherwise lat/lon.
        Returns:
            stations: Nx2 numpy array of station coordinates.
        """
        if not transform:
            stations = self.df[["xr", "yr"]]
        else:
            stations = self.df[["lonr", "latr"]]
        return stations.to_numpy(dtype=float)

    def _get_receivers(self, source_index):
        """
        For a given source_index, return the (yr, xr) coordinates of all receivers.
        Args:
            source_index: Integer index of source_id in sorted unique IDs.
        Returns:
            recv_coords: Mx2 array of (yr, xr) in kilometers.
        """
        subset = self.df[self.df["source_id"] == source_index]
        return subset[["yr", "xr"]].to_numpy(dtype=float)

    def solve(self, return_gradient=False, nsweep=2):
        """
        Run the underlying Eikonal2D solver on all unique sources.

        Args:
            return_gradient: If True, return both travel times and gradient.
            nsweep: Number of sweeps for the solver.
        Returns:
            List of "grid" callables (one per source).
        """
        logger.debug(
            "Solving Eikonal for %d unique sources",
            len(pd.unique(self.df["source_id"])),
        )
        sources = self._get_sources()
        self.tt = Eikonal2D.solve(
            self, sources=sources, nsweep=nsweep, return_gradient=return_gradient
        )
        logger.debug("Eikonal solve returned %d traveltime grids", len(self.tt))
        return self.tt

    def calculate_traveltimes(self):
        """
        Evaluate each travel-time field at its receivers, storing results in self.traveltimes.
        """
        logger.debug("Calculating traveltimes at %d records", len(self.df))
        offset = 0
        for i, grid in enumerate(self.tt):
            points = self._get_receivers(i)
            n = points.shape[0]
            self.traveltimes[offset : n + offset] = grid(points)
            offset += n

    def get_residuals(self):
        """
        Compute residuals = observed_tt - predicted_tt.

        Returns:
            1D array of length N (rows in df).
        """
        return self.df["tt"].to_numpy() - self.traveltimes

    def calc_residuals(self):
        """
        Compute cost = (residual^T * Cd * residual) / N.

        Returns:
            Scalar misfit value.
        """
        res = self.get_residuals()
        return float((res @ self.Cd @ res) / len(res))

    def save_measurements(self, filename, field_name="tt"):
        """
        Write predicted travel times back into self.df['tt'] and overwrite CSV.

        Args:
            filename: Path to output CSV.
            field_name: Column name to store the traveltime values.
        """
        self.df[field_name] = self.traveltimes
        logger.info(
            "Saving measurements (with predicted traveltimes in column %s) to %s",
            field_name,
            filename,
        )
        io.save_measurements_csv(self.df, filename)

    def save_model_csv(self, filename):
        np.savetxt(filename, self._grid, fmt="%.2f", delimiter=",")

    def save_model_xyz(self, filename, use_latlon=False):
        flat = self._grid.flatten()
        idx = 0
        with open(filename, "w") as f:
            for y in self.zaxis:
                for x in self.xaxis:
                    if use_latlon:
                        lon, lat = self.inv_transformer.transform(
                            x * 1000 + self.base_xy[0],
                            y * 1000 + self.base_xy[1],
                        )
                        f.write(f"{lon:.3f} {lat:.3f} {flat[idx]:.3f}\n")
                    else:
                        f.write(f"{x:.3f} {y:.3f} {flat[idx]:.3f}\n")
                    idx += 1

    def add_noise(self, sigma, mu=0):
        """
        Add Gaussian noise (mean=mu, std=sigma) to predicted travel times.

        Args:
            sigma: Standard deviation of noise.
            mu: Mean of noise (default: 0).
        """
        logger.info(
            "Adding Gaussian noise (mu=%.3f, sigma=%.3f) to traveltimes", mu, sigma
        )
        noise = np.random.normal(mu, sigma, size=(self.traveltimes.size))
        self.traveltimes += noise

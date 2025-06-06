from fteikpy import Eikonal2D
import logging
import pandas as pd
import numpy as np
import pyproj
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
        logger.debug("Initializing EikonalSolver with grid shape %s, gridsize %s", grid.shape, gridsize)
        # Load measurements
        self.df = io.load_measurements_csv(measurements_csv)
        logger.info("Loaded %d measurement rows from %s", len(self.df), measurements_csv)
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
        logger.info("Transformers initialized for LAEA centered at (%.4f, %.4f)", lon, lat)


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
        self.base = self.transformer.transform(self.base[0], self.base[1])
        Es, Ns = self.transformer.transform(self.df["lons"], self.df["lats"])
        Er, Nr = self.transformer.transform(self.df["lonr"], self.df["latr"])
        self.df["xs"] = (Es - self.base[0]) / 1000
        self.df["ys"] = (Ns - self.base[1]) / 1000
        self.df["xr"] = (Er - self.base[0]) / 1000
        self.df["yr"] = (Nr - self.base[1]) / 1000

    def transform_to_latlon(self, X, Y):
        """
        Convert local (x, y) in kilometers back to (lon, lat).
        Requires transform_to_xy() to have been called first.

        Args:
            x_km: 1D array of x coordinates (km).
            y_km: 1D array of y coordinates (km).
        Returns:
            lons, lats as 1D arrays.
        """
        logger.info("Transforming local XY (km) to lat/lon")
        if self.inv_transformer is None:
            raise RuntimeError(
                "Inverse transformer not initialized. Call transform_to_xy() first."
            )

        lons, lats = self.inv_transformer.transform(
            (X * 1000 + self.base[0]), (Y * 1000 + self.base[1])
        )
        return lons, lats

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

    def _load_measurements(self, filename):
        """
        Read the measurement CSV into a DataFrame.
        Expects at least:
          - source_id (uint32)
          - either (xs, ys, xr, yr) OR (lons, lats, lonr, latr)
          - optional 'tt' (observed travel time) and 'sigma'.

        Args:
            filename: Path to CSV file.
        Returns:
            df: pandas DataFrame with measurement data.
        """
        df = pd.read_csv(filepath_or_buffer=filename, dtype={"source_id": np.uint32})
        input_validator.validate_measurement_csv(df)
        return df

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
        If use_latlon=False, returns (xr, yr); else returns (lonr, latr).

        Args:
            use_latlon: Bool flag. If False, return local XY; otherwise lat/lon.
        Returns:
            stations: Nx2 numpy array of station coordinates.
        """
        if not use_latlon:
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
        logger.info("Solving Eikonal for %d unique sources", len(pd.unique(self.df["source_id"])))
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
        logger.info("Calculating traveltimes at %d records", len(self.df))
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

    def save_measurements(self, filename):
        """
        Write predicted travel times back into self.df['tt'] and overwrite CSV.

        Args:
            filename: Path to output CSV.
        """
        self.df["tt"] = self.traveltimes
        logger.info("Saving measurements (with predicted tt) to %s", filename)
        io.save_measurements_csv(self.df, filename)

    def add_noise(self, sigma, mu=0):
        """
        Add Gaussian noise (mean=mu, std=sigma) to predicted travel times.

        Args:
            sigma: Standard deviation of noise.
            mu: Mean of noise (default: 0).
        """
        logger.info("Adding Gaussian noise (mu=%.3f, sigma=%.3f) to traveltimes", mu, sigma)
        noise = np.random.normal(mu, sigma, size=(self.traveltimes.size))
        self.traveltimes += noise

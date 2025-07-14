import numpy as np
import pandas as pd
import pytest

from tomoFMpy.core.solver import Eikonal_Solver


# Monkey‐patch fteikpy.Eikonal2D.solve so it returns a dummy grid that yields constant travel times
@pytest.fixture(autouse=True)
def patch_eikonal2d(monkeypatch):
    """
    Replace Eikonal2D.solve with a dummy implementation that returns a single "grid" callable
    which returns 0.5 for any input points. This allows testing solve()/calculate_traveltimes()
    without requiring a real Eikonal solver.
    """

    def dummy_solve(self, sources=None, nsweep=None, return_gradient=False):
        class DummyGrid:
            def __call__(self, points):
                # Return 0.5 for each receiver point
                return np.full(len(points), 0.5, dtype=float)

        # Assume exactly one unique source for simplicity
        return [DummyGrid()]

    import fteikpy

    monkeypatch.setattr(fteikpy.Eikonal2D, "solve", dummy_solve)
    yield


@pytest.fixture
def simple_csv(tmp_path):
    """
    Create a small CSV with two receivers from a single source:
    columns: source_id, lons, lats, lonr, latr, tt, sigma
    """
    df = pd.DataFrame(
        {
            "source_id": [0, 0],
            "lons": [10.0, 10.0],
            "lats": [50.0, 50.0],
            "lonr": [10.1, 10.0],
            "latr": [50.0, 50.1],
            "tt": [1.0, 2.0],
            "sigma": [2.0, 0.5],
        }
    )
    path = tmp_path / "measurements.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_measurements_and_covariance(simple_csv):
    grid = np.zeros((2, 2))

    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=simple_csv,
        origin=None,
        bl_corner=(10.0, 50.0),
    )

    # Check that df was loaded correctly
    df_loaded = solver.df
    assert "source_id" in df_loaded.columns
    assert df_loaded.shape == (
        2,
        7,
    )  # 7 columns: source_id, lons, lats, lonr, latr, tt, sigma

    # Covariance diagonal should be [1/2.0, 1/0.5] = [0.5, 2.0]
    expected_sigma = np.array([0.5, 2.0], dtype=float)
    diag = np.diag(solver.Cd)
    assert np.allclose(diag, expected_sigma)

    # Off‐diagonals must be zero
    off_diag = solver.Cd.copy()
    np.fill_diagonal(off_diag, 0.0)
    assert np.allclose(off_diag, 0.0)


def test_transform_and_inverse_roundtrip(simple_csv):
    grid = np.zeros((5, 5))
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=simple_csv,
        bl_corner=(10.0, 50.0),
    )

    # Before transforming, df should have lons/lats or lonr/latr
    for col in ("lons", "lats", "lonr", "latr"):
        assert col in solver.df.columns

    # Before transforming, df should not have xs/ys or xr/yr
    for col in ("xs", "ys", "xr", "yr"):
        assert col not in solver.df.columns

    # Perform transform to local x/y
    solver.transform_to_xy()

    # Now df should contain xs, ys, xr, yr
    for col in ("xs", "ys", "xr", "yr"):
        assert col in solver.df.columns

    orig = solver.df.copy()

    # Run inverse transform
    solver.transform_to_latlon()
    assert np.allclose(orig, solver.df, atol=10e-4)

    solver.transform_to_xy()
    solver.transform_to_latlon()

    assert np.allclose(orig, solver.df, atol=10e-4)


def test_get_sources_and_receivers_after_transform(simple_csv):
    grid = np.ones((5, 5))
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=simple_csv,
        bl_corner=(10.0, 50.0),
    )
    solver.transform_to_xy()

    # _get_sources should return one source (ys, xs)
    sources = solver._get_sources()
    assert sources.shape == (1, 2)
    # All y and x for source 0 are identical (first row)
    assert np.allclose(sources[0], [solver.df["ys"].iloc[0], solver.df["xs"].iloc[0]])

    # _get_receivers for source 0 should return two points (yr, xr)
    recs = solver._get_receivers(0)
    # Should have shape (2, 2)
    assert recs.shape == (2, 2)
    # Entries must match the (yr, xr) columns
    expected = solver.df.loc[solver.df["source_id"] == 0, ["yr", "xr"]].to_numpy()
    assert np.allclose(recs, expected)


def test_solve_and_traveltimes_and_residuals(simple_csv):
    grid = np.zeros((3, 3))
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=simple_csv,
        bl_corner=(10.0, 50.0),
    )
    solver.transform_to_xy()

    assert np.allclose(solver.traveltimes, 0.0)

    tt_list = solver.solve(return_gradient=False, nsweep=1)
    assert isinstance(tt_list, list) and len(tt_list) == 1

    solver.calculate_traveltimes()
    assert np.allclose(solver.traveltimes, 0.5)

    res = solver.get_residuals()
    assert np.allclose(res, [0.5, 1.5])

    cost = solver.calc_residuals()
    assert pytest.approx(cost, rel=1e-8) == 2.3125


def test_save_and_reload_measurements(tmp_path, simple_csv):
    # Copy CSV into tmp_path to avoid overwriting fixture
    dest = tmp_path / "copy.csv"
    import shutil

    shutil.copy(simple_csv, dest)

    grid = np.zeros((2, 2))
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=str(dest),
        bl_corner=(10.0, 50.0),
    )
    solver.transform_to_xy()

    # Manually set traveltimes
    solver.traveltimes = np.array([0.2, 0.4], dtype=float)
    solver.save_measurements(str(dest))

    # Reload and confirm the file was overwritten
    df2 = pd.read_csv(dest)
    assert "tt" in df2.columns
    assert np.allclose(df2["tt"].to_numpy(), [0.2, 0.4])


def test_save_model_xyz_no_latlon(tmp_path, simple_csv):
    rf = tmp_path / "out7"
    grid = np.array([[1.23, 2.00], [1, 1]])
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=simple_csv,
        bl_corner=(10.0, 50.0),
    )
    solver.save_model_xyz(tmp_path / "3.xyz")
    lines = (tmp_path / "3.xyz").read_text().splitlines()
    assert len(lines) == 4
    assert lines[0] == "0.000 0.000 1.230"
    assert lines[1] == "1.000 0.000 2.000"
    assert lines[2] == "0.000 1.000 1.000"
    assert lines[3] == "1.000 1.000 1.000"


def test_save_model_csv(tmp_path, simple_csv):
    rf = tmp_path / "5.csv"
    grid = np.array([[1.23, 2.00], [1, 1]])
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=simple_csv,
        bl_corner=(10.0, 50.0),
    )
    solver.save_model_csv(rf)
    loaded = np.loadtxt(rf, delimiter=",")
    assert np.allclose(loaded, grid)


def test_add_noise_affects_traveltimes(simple_csv, monkeypatch):
    grid = np.zeros((3, 3))
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=simple_csv,
        bl_corner=(10.0, 50.0),
    )
    solver.transform_to_xy()

    solver.traveltimes = np.array([0.5, 0.5], dtype=float)

    solver.add_noise(sigma=0.0, mu=0.0)
    assert np.allclose(solver.traveltimes, [0.5, 0.5])

    def fake_normal(mu, sigma, size=None):
        return np.array([0.1, -0.2], dtype=float)

    monkeypatch.setattr(np.random, "normal", fake_normal)

    solver.traveltimes = np.array([0.5, 0.5], dtype=float)
    solver.add_noise(sigma=0.1, mu=0.0)
    assert solver.traveltimes.shape == (2,)
    assert np.allclose(solver.traveltimes, [0.6, 0.3])


@pytest.fixture
def csv_without_sigma(tmp_path):
    """
    Create a CSV that omits the sigma column entirely.
    """
    df = pd.DataFrame(
        {
            "source_id": [0],
            "lons": [10.0],
            "lats": [50.0],
            "lonr": [10.1],
            "latr": [50.1],
            "tt": [1.0],
            # no sigma
        }
    )
    path = tmp_path / "nosigma.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_covariance_defaults_to_identity(csv_without_sigma):
    grid = np.zeros((2, 2))
    solver = Eikonal_Solver(
        grid=grid,
        gridsize=(1.0, 1.0),
        measurements_csv=csv_without_sigma,
        bl_corner=(10.0, 50.0),
    )

    # Since there was no sigma column, Cd should be identity (size=1×1)
    assert solver.Cd.shape == (1, 1)
    assert pytest.approx(solver.Cd[0, 0], rel=1e-8) == 1.0


if __name__ == "__main__":
    pytest.main()

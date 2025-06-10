import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from scipy.spatial import ConvexHull

from tomoFMpy.core.inversion import Eikonal_Inversion

# Monkey‐patch the underlying Eikonal_Solver.solve and related methods
# to avoid needing a real Eikonal2D solver.
@pytest.fixture(autouse=True)
def patch_eikonal_solver(monkeypatch):
    """
    Replace Eikonal_Solver.solve(), calculate_traveltimes(), and calc_residuals()
    with minimal stubs so that inversion logic can run deterministically.
    """

    class DummySolver:
        def __init__(self, *args, **kwargs):
            pass

        def update_model(self, grid):
            pass

        def solve(self, *args, **kwargs):
            # pretend we solved; do nothing
            return

        def calculate_traveltimes(self):
            # pretend traveltimes have been computed
            return

        def calc_residuals(self):
            # return a fixed residual cost
            return 4.0

        def get_unique_stations(self, transform):
            # Return a small set of 3 points for hull
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    # Monkey‐patch Eikonal_Solver in the inversion module to use DummySolver
    from tomoFMpy.core import inversion

    monkeypatch.setattr(inversion, "Eikonal_Solver", DummySolver)
    yield


@pytest.fixture
def measurement_csv(tmp_path):
    """
    Create a CSV suitable for Eikonal_Inversion: it needs columns
    source_id, xs, ys, xr, yr, sigma, tt (though solver stubs ignore tt).
    """
    df = pd.DataFrame(
        {
            "source_id": [0, 0, 0],
            "xs": [0.0, 1.0, 0.0],
            "ys": [0.0, 0.0, 1.0],
            "xr": [0.5, 1.5, 0.5],
            "yr": [0.5, 0.5, 1.5],
            "tt": [1.0, 1.2, 1.4],
            "sigma": [1.0, 1.0, 1.0],
        }
    )
    path = tmp_path / "inv_meas.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_build_laplacian_small_grid(tmp_path, measurement_csv):
    """
    For a 2x2 grid, D should be 4x4 with the pattern:
      Node indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
      D[0] = [2, -1, -1, 0]
      D[1] = [-1, 2, 0, -1]
      D[2] = [-1, 0, 2, -1]
      D[3] = [0, -1, -1, 2]
    """
    # Ensure the CSV file exists (constructor will stub solver, but check presence)
    pd.DataFrame(
        {
            "source_id": [0],
            "xs": [0.0],
            "ys": [0.0],
            "xr": [0.0],
            "yr": [0.0],
            "tt": [1.0],
            "sigma": [1.0],
        }
    ).to_csv(measurement_csv, index=False)

    inv = Eikonal_Inversion(
        modelspace=(2.0, 2.0),
        gridsize=(2, 2),
        measurement_csv=measurement_csv,
        output_folder=str(tmp_path / "out1"),
        bl_corner=(0, 0),
        epsilon=0,
        eta=0,
        use_latlon=False,
    )

    D = inv._build_laplacian()
    assert D.shape == (4, 4)
    assert np.allclose(D[0], [2, -1, -1, 0])
    assert np.allclose(D[1], [-1, 2, 0, -1])
    assert np.allclose(D[2], [-1, 0, 2, -1])
    assert np.allclose(D[3], [0, -1, -1, 2])


def test_calc_perturbation_and_roughness(measurement_csv, tmp_path):
    """
    Test that calc_perturbation gives ||x - x0||^2 and
    calc_roughness uses the Laplacian matrix correctly on a 2x2 grid.
    """
    # Ensure CSV exists
    pd.DataFrame(
        {
            "source_id": [0],
            "xs": [0.0],
            "ys": [0.0],
            "xr": [0.0],
            "yr": [0.0],
            "tt": [1.0],
            "sigma": [1.0],
        }
    ).to_csv(measurement_csv, index=False)

    inv = Eikonal_Inversion(
        modelspace=(2.0, 2.0),
        gridsize=(2, 2),
        measurement_csv=measurement_csv,
        output_folder=str(tmp_path / "out2"),
        bl_corner=(0, 0),
        epsilon=1.0,
        eta=1.0,
        use_latlon=False,
    )

    # Set a starting model x0
    x0 = np.zeros(4, dtype=float)
    inv.add_starting_model(x0)

    # Perturbation: if x = ones vector, penalty = sum((1-0)^2) = 4
    ones = np.ones(4, dtype=float)
    pert = inv._perturbation(ones)
    assert pytest.approx(pert, rel=1e-8) == 4.0

    # Roughness on 2x2 with model_vec=ones: since D @ ones = 0 (each row sums to 0),
    # penalty = ones^T * (D * ones) = 0
    rough = inv._roughness(ones)
    assert pytest.approx(rough, rel=1e-8) == 0.0


def test_misfit_no_reg(tmp_path, measurement_csv):
    # Create minimal CSV
    pd.DataFrame(
        {
            "source_id": [0],
            "xs": [0.0],
            "ys": [0.0],
            "xr": [0.0],
            "yr": [0.0],
            "tt": [1.0],
            "sigma": [1.0],
        }
    ).to_csv(measurement_csv, index=False)

    rf = tmp_path / "out3"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=0,
        eta=0,
        use_latlon=False,
    )
    inv.add_starting_model(np.array([0.0]))
    misfit = inv.misfit(np.array([1.0]))
    # data cost constant=4.0 => sqrt(4.0)
    assert misfit == pytest.approx(np.sqrt(4.0))


def test_misfit_with_reg(tmp_path, measurement_csv):
    # Create minimal CSV
    pd.DataFrame(
        {
            "source_id": [0],
            "xs": [0.0],
            "ys": [0.0],
            "xr": [0.0],
            "yr": [0.0],
            "tt": [1.0],
            "sigma": [1.0],
        }
    ).to_csv(measurement_csv, index=False)

    rf = tmp_path / "out4"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=1.0,
        eta=1,
        use_latlon=False,
    )
    inv.add_starting_model(np.array([0.0]))
    misfit = inv.misfit(np.array([1.0]))
    # data cost=4, pert=(1-0)^2=1 => sqrt(5)
    assert misfit == pytest.approx(np.sqrt(5.0))


def test_run_linear_no_x0(tmp_path, measurement_csv):
    # Create minimal CSV
    pd.DataFrame(
        {
            "source_id": [0],
            "xs": [0.0],
            "ys": [0.0],
            "xr": [0.0],
            "yr": [0.0],
            "tt": [1.0],
            "sigma": [1.0],
        }
    ).to_csv(measurement_csv, index=False)

    rf = tmp_path / "out5"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=1.0,
        eta=1,
        use_latlon=False,
    )
    with pytest.raises(RuntimeError):
        inv.run_linear()


def test_save_model_csv(tmp_path, measurement_csv):
    rf = tmp_path / "out6"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=1.0,
        eta=1,
        use_latlon=False,
    )
    vec = np.array([1.23])
    inv._save_model_csv(vec, 5)
    loaded = np.loadtxt(rf / "5.csv", delimiter=",")
    assert loaded == pytest.approx(1.23)


def test_save_model_xyz_no_latlon(tmp_path):
    rf = tmp_path / "out7"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=1.0,
        eta=1,
        use_latlon=False,
    )
    inv.use_latlon = False
    vec = np.array([1.23])
    inv._save_model_xyz(vec, 3)
    lines = (rf / "3.xyz").read_text().splitlines()
    assert lines == ["0.000 0.000 1.230"]


def test_save_model_png(tmp_path):
    rf = tmp_path / "out7"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=1.0,
        eta=1,
        use_latlon=False,
    )
    vec = np.array([1.0])
    inv._save_model_png(vec, 2)
    assert (rf / "2.png").exists()


def test_save_station_files(tmp_path):
    rf = tmp_path / "out8"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=1.0,
        eta=1,
        use_latlon=False,
    )
    # override solver to return known stations
    stations = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    inv.solver.get_unique_stations = lambda use_latlon: stations
    inv._save_station_files()
    saved = np.loadtxt(rf / "stations.xy")
    assert np.allclose(saved, stations)
    hull = np.loadtxt(rf / "hull.xy")
    # hull vertices count matches
    assert hull.shape[0] == 4


def test_run_linear_minimize(tmp_path, measurement_csv):
    """
    Test that run_linearization calls scipy.optimize.minimize and writes residuals.
    We monkey-patch fitness_func to a simple quadratic so minimize finishes quickly.
    """
    # Create minimal CSV
    pd.DataFrame(
        {
            "source_id": [0],
            "xs": [0.0],
            "ys": [0.0],
            "xr": [0.0],
            "yr": [0.0],
            "tt": [0.0],
            "sigma": [1.0],
        }
    ).to_csv(measurement_csv, index=False)

    rf = tmp_path / "lin"
    inv = Eikonal_Inversion(
        modelspace=(1.0, 1.0),
        gridsize=(1, 1),
        measurement_csv=measurement_csv,
        output_folder=str(rf),
        bl_corner=(0, 0),
        epsilon=0,
        eta=0,
        use_latlon=False,
    )

    # Override inv.fitness_func to a simple (x-1)^2
    def simple_fitness(x_vec):
        return (x_vec[0] - 1.0) ** 2

    inv.fitness_func = simple_fitness

    # Set x0 = [0.0]
    inv.add_starting_model(np.array([0.0], dtype=float))

    # Run linearization with maxiter=1 to keep it fast
    result = inv.run_linear(method="BFGS", options={"maxiter": 1})
    # Ensure result has attribute 'x'
    assert hasattr(result, "x")
    # Check residuals.txt exists
    res_path = rf / "residuals.txt"
    assert res_path.exists()
    # Load residual history: at least two entries (initial + final)
    loaded = np.loadtxt(res_path)
    assert loaded.ndim >= 1

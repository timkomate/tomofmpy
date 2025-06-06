import pandas as pd
import numpy as np
import pytest
import os

from tomoFMpy.utils.io import (
    validate_measurement_csv,
    load_measurements_csv,
    save_measurements_csv,
)


def test_validate_success_latlon(caplog):
    # DataFrame with lat/lon columns, no 'tt'. Should warn but not raise
    df = pd.DataFrame(
        {
            "source_id": [0, 1],
            "lons": [10.0, 20.0],
            "lats": [50.0, 60.0],
            "lonr": [10.1, 20.1],
            "latr": [50.1, 60.1],
            "sigma": [1.0, 2.0],
        }
    )
    validate_measurement_csv(df)
    assert "Warning: CSV has no 'tt' column" in caplog.text


def test_validate_success_xy(caplog):
    # DataFrame with xs/ys/xr/yr and 'tt' -> no warning, no error
    df = pd.DataFrame(
        {
            "source_id": [0, 0],
            "xs": [1.0, 2.0],
            "ys": [1.0, 2.0],
            "xr": [3.0, 4.0],
            "yr": [3.0, 4.0],
            "tt": [1.5, 2.5],
        }
    )
    validate_measurement_csv(df)
    assert caplog.text == ""


def test_validate_missing_columns():
    # Missing both lat/lon and XY sets -> must raise
    df = pd.DataFrame(
        {
            "source_id": [0],
            "tt": [1.0],
        }
    )
    with pytest.raises(ValueError) as exc:
        validate_measurement_csv(df)
    assert "must contain either (xs, ys, xr, yr) or (lons, lats, lonr, latr)" in str(
        exc.value
    )


def test_validate_sigma_nonpositive():
    # DataFrame where sigma contains zero or negative -> must raise
    df = pd.DataFrame(
        {
            "source_id": [0, 1],
            "xs": [0.0, 1.0],
            "ys": [0.0, 1.0],
            "xr": [1.0, 2.0],
            "yr": [1.0, 2.0],
            "tt": [1.0, 2.0],
            "sigma": [1.0, 0.0],
        }
    )
    with pytest.raises(ValueError) as exc:
        validate_measurement_csv(df)
    assert "All 'sigma' values must be > 0." in str(exc.value)


def test_load_measurements_csv_success(tmp_path):
    # Write a valid CSV (using local XY) and then load it
    csv_path = tmp_path / "meas.csv"
    df_original = pd.DataFrame(
        {
            "source_id": [0, 0],
            "xs": [0.5, 1.5],
            "ys": [0.5, 1.5],
            "xr": [1.0, 2.0],
            "yr": [1.0, 2.0],
            "tt": [1.0, 2.0],
            "sigma": [1.0, 2.0],
        }
    )
    df_original["source_id"] = df_original["source_id"].astype(np.uint32)
    df_original.to_csv(csv_path, index=False)

    df_loaded = load_measurements_csv(str(csv_path))
    pd.testing.assert_frame_equal(df_loaded, df_original)


def test_load_measurements_csv_invalid(tmp_path):
    # Write a CSV missing both XY and lat/lon -> load must raise ValueError
    csv_path = tmp_path / "bad.csv"
    df_bad = pd.DataFrame(
        {
            "source_id": [0],
            "tt": [1.0],
        }
    )
    df_bad.to_csv(csv_path, index=False)

    with pytest.raises(ValueError) as exc:
        load_measurements_csv(str(csv_path))
    assert "must contain either (xs, ys, xr, yr) or (lons, lats, lonr, latr)" in str(
        exc.value
    )


def test_save_measurements_csv(tmp_path):
    # Create a DataFrame, save it, then reload to ensure content matches
    df = pd.DataFrame(
        {
            "source_id": [0, 1],
            "xs": [0.0, 1.0],
            "ys": [0.0, 1.0],
            "xr": [1.0, 2.0],
            "yr": [1.0, 2.0],
            "tt": [1.0, 2.0],
            "sigma": [1.0, 1.0],
        }
    )
    df["source_id"] = df["source_id"].astype(np.uint32)
    out_path = tmp_path / "out.csv"
    save_measurements_csv(df, str(out_path))

    df_reloaded = pd.read_csv(out_path, dtype={"source_id": np.uint32})
    pd.testing.assert_frame_equal(df_reloaded, df)

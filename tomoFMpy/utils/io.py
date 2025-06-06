import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_measurement_csv(df):
    required = ["source_id"]
    if {"lons", "lats", "lonr", "latr"}.issubset(df.columns):
        pass
    elif {"xs", "ys", "xr", "yr"}.issubset(df.columns):
        pass
    else:
        raise ValueError(
            "Measurement CSV must contain either (xs, ys, xr, yr) or (lons, lats, lonr, latr)."
        )

    if "tt" not in df.columns:
        # Forward‐solve only; we’ll write tt out later—but warn user if they intend inversion.
        logging.warning("Warning: CSV has no 'tt' column—forward. Solving only.")

    # If sigma exists, ensure no zeros
    if "sigma" in df.columns and (df["sigma"] <= 0).any():
        raise ValueError("All 'sigma' values must be > 0.")



def load_measurements_csv(path):
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
    df = pd.read_csv(path, dtype={"source_id": np.uint32})
    validate_measurement_csv(df)
    return df


def save_measurements_csv(df, path):
    """
    Write predicted travel times back into self.df['tt'] and overwrite CSV.

    Args:
        filename: Path to output CSV.
    """
    df.to_csv(path, index=False)

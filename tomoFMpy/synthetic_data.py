import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageOps
from tomoFMpy.core import solver
from tomoFMpy.utils.parameter_init import Config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def checkerboard(shape, tile_size):
    """
    Generate a checkerboard pattern with the specified shape and tile size.

    Args:
        shape (tuple): Shape of the checkerboard grid.
        tile_size (int): Size of each tile in the checkerboard.

    Returns:
        ndarray: Checkerboard pattern.

    """
    logger.debug(
        "Generating checkerboard pattern with shape %s and tile size %s",
        shape,
        tile_size,
    )
    idx = np.indices(shape)
    return ((idx // tile_size).sum(axis=0)) % 2


def read_image(filepath, dv, v0):
    """
    Read an image file and convert it into a velocity model.

    Args:
        filepath (str): Path to the image file.
        dv (float): Velocity increment (maps 0–255 to 0–dv).
        v0 (float): Base velocity (added to scaled pixel values).

    Returns:
        ndarray: Velocity model derived from the image.

    """
    logger.info("Reading image file %s", filepath)
    try:
        im = PIL.Image.open(filepath)
    except FileNotFoundError:
        logger.error("Image file not found: %s", filepath)
        raise
    gray_image = PIL.ImageOps.grayscale(im)
    # Scale to [v0, v0+dv] and flip vertically
    velocity_model = ((np.array(gray_image) / 255) * dv) + v0
    velocity_model = np.flipud(velocity_model)
    return velocity_model


def build_source_boundary(config):
    """
    Create source coordinates along the rectangular boundary defined in config.

    Args:
        config (Config): Configuration object with xmin, xmax, ymin, ymax, nsrc.

    Returns:
        np.ndarray: Array of shape (N, 2) of (x, y) source positions.
    """
    # Points along each edge (excluding duplicates at corners will be removed later)
    x_lin = np.linspace(config.xmin, config.xmax, config.nsrc)
    y_lin = np.linspace(config.ymin, config.ymax, config.nsrc)

    # Bottom edge: y = ymin, x from xmin to xmax
    bottom = np.column_stack([x_lin, np.full(config.nsrc, config.ymin)])
    # Right edge: x = xmax, y from ymin to ymax
    right = np.column_stack([np.full(config.nsrc, config.xmax), y_lin])
    # Top edge: y = ymax, x from xmax down to xmin
    top = np.column_stack([x_lin[::-1], np.full(config.nsrc, config.ymax)])
    # Left edge: x = xmin, y from ymax down to ymin
    left = np.column_stack([np.full(config.nsrc, config.xmin), y_lin[::-1]])

    boundary = np.vstack([bottom, right, top, left])
    # Remove duplicate corner points
    unique_boundary = np.unique(boundary, axis=0)
    return unique_boundary


def random_geometry(config):
    """
    Generate random source-receiver geometry based on the provided configuration.

    Args:
        config (Config): Configuration object.

    Returns:
        DataFrame: Random geometry data.

    """
    np.random.seed(config.seed)

    xr = np.random.uniform(config.xmin, config.xmax, config.r)
    yr = np.random.uniform(config.ymin, config.ymax, config.r)
    receivers = np.column_stack([xr, yr])

    sources = build_source_boundary(config)
    num_sources = sources.shape[0]

    total_pairs = num_sources * config.r
    data = np.zeros((total_pairs, 6), dtype=float)
    idx = 0
    for sid, (sx, sy) in enumerate(sources):
        for rx, ry in receivers:
            data[idx] = [sid, sx, sy, rx, ry, 1.0]
            idx += 1
    if config.latlon:
        columns = ["source_id", "lons", "lats", "lonr", "latr", "sigma"]
    else:
        columns = ["source_id", "xs", "ys", "xr", "yr", "sigma"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(config.fname, index=False)
    logger.info("Saved geometry to %s", config.fname)
    return df


def plot_eikonal_results(eik, config: Config) -> None:
    """
    If enabled, plot the speed grid and ray paths/scatter of sources and receivers.

    Args:
        eik: An instance of tomo_eikonal.Eikonal_Solver after running solve().
        config (Config): Configuration object with xmin, xmax, ymin, ymax.
    """
    logger.info("Plotting eikonal output")
    fig, ax = plt.subplots()
    mesh = ax.pcolor(eik.xaxis, eik.zaxis, eik.grid)
    plt.colorbar(mesh, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    plt.title("Velocity Grid (Eikonal Solver)")
    plt.show()

    fig, ax = plt.subplots()
    mesh = ax.pcolor(eik.xaxis, eik.zaxis, eik.grid, shading="auto")
    plt.colorbar(mesh, ax=ax)
    # Scatter source and receiver locations
    df = eik.df
    if config.latlon:
        ax.scatter(df["lons"], df["lats"], marker="h", c="k", label="Sources")
        ax.scatter(df["lonr"], df["latr"], marker="^", c="r", label="Receivers")
    else:
        ax.scatter(df["xs"], df["ys"], marker="h", c="k", label="Sources")
        ax.scatter(df["xr"], df["yr"], marker="^", c="r", label="Receivers")
    ax.set_xlim(config.xmin, config.xmax)
    ax.set_ylim(config.ymin, config.ymax)
    plt.title("Sources (black) and Receivers (red)")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Run the Eikonal Solver with synthetic velocity models."
    )

    # Add the command-line arguments
    parser.add_argument(
        "--config",
        help="Path to the config file",
        default="configs/synthetic_config.ini",
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    logger.info("Using config file: %s", args.config)

    # Parse the config file
    config = Config(args.config)
    logger.debug(config)

    # Generate velocity model
    logger.info("Generating velocity model using method: %s", config.method)
    if config.method == "checkerboard":
        velocity_model = (
            checkerboard((config.x, config.y), config.tile_size) * config.dv + config.v0
        )
    elif config.method == "image":
        velocity_model = read_image(config.image_path, config.dv, config.v0)
    else:
        raise ValueError(
            f"Invalid method '{config.method}' specified in the config file."
        )

    # Generate random geometry
    logger.info("Generating random geometry")
    geometry_df = random_geometry(config)

    ny, nx = velocity_model.shape
    dx = config.x / nx
    dy = config.y / ny

    # Solve Eikonal at source
    eik = solver.Eikonal_Solver(
        velocity_model,
        gridsize=(dy, dx),
        measurements_csv=config.fname,
        bl_corner=(config.bl_lon, config.bl_lat),
    )
    if config.latlon:
        eik.transform_to_xy()

    if config.plot:
        plot_eikonal_results(eik, config)

    eik.solve()
    eik.calculate_traveltimes()
    eik.add_noise(config.noise)
    eik.save_measurements(config.fname)
    logger.info("Done")

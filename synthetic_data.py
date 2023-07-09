import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.tomo_eikonal as tomo_eikonal
import PIL.Image
import PIL.ImageOps
import argparse
from utils.parameter_init import Config


def checkerboard(shape, tile_size):
    """
    Generate a checkerboard pattern with the specified shape and tile size.

    Args:
        shape (tuple): Shape of the checkerboard grid.
        tile_size (int): Size of each tile in the checkerboard.

    Returns:
        ndarray: Checkerboard pattern.

    """
    return (np.indices(shape) // tile_size).sum(axis=0) % 2


def read_image(filepath, dv, v0):
    """
    Read an image file and convert it into a velocity model.

    Args:
        filepath (str): Path to the image file.
        dv (float): Velocity increment.
        v0 (float): Starting velocity.

    Returns:
        ndarray: Velocity model derived from the image.

    """
    im = PIL.Image.open(filepath)
    gray_image = PIL.ImageOps.grayscale(im)
    velocity_model = ((np.array(gray_image) / 255) * dv) + v0
    velocity_model = np.flipud(velocity_model)
    return velocity_model


def random_geometry(config):
    """
    Generate random geometry based on the provided configuration.

    Args:
        config (Config): Configuration object.

    Returns:
        DataFrame: Random geometry data.

    """
    np.random.seed(config.seed)

    xr = np.random.uniform(config.xmin, config.xmax, config.r)
    yr = np.random.uniform(config.ymin, config.ymax, config.r)
    coord_rec = np.vstack([xr, yr]).T

    xs = np.linspace(config.ymin, config.ymax, config.nsrc)
    ys = np.linspace(config.xmin, config.xmax, config.nsrc)
    coords1 = np.column_stack([ys, np.full_like(ys, config.ymin)])
    coords2 = np.column_stack([np.full_like(xs, config.xmax), xs])
    coords3 = np.column_stack([ys, np.full_like(ys, config.ymax)])
    coords4 = np.column_stack([np.full_like(xs, config.xmin), xs])
    coords_source = np.vstack([coords1, coords2, coords3, coords4])
    coords_source = np.unique(coords_source, axis=0)
    n = coords_source.shape[0]

    data = np.zeros((n * config.r, 6))
    c = 0
    for i, source in enumerate(coords_source):
        for j, rec in enumerate(coord_rec):
            data[c] = [i, source[0], source[1], rec[0], rec[1], config.sigma]
            c += 1
    if config.latlon:
        df = pd.DataFrame(
            data, columns=["source_id", "lons", "lats", "lonr", "latr", "sigma"]
        )
    else:
        df = pd.DataFrame(
            data, columns=["source_id", "xs", "ys", "xr", "yr", "sigma"]
        )
    df.to_csv(path_or_buf=config.fname, index=False)
    if config.plot:
        plt.scatter(coords1[:, 0], coords1[:, 1])
        plt.scatter(coords2[:, 0], coords2[:, 1])
        plt.scatter(coords3[:, 0], coords3[:, 1])
        plt.scatter(coords4[:, 0], coords4[:, 1])
        plt.xlim([config.xmin, config.xmax])
        plt.ylim([config.ymin, config.ymax])
        plt.show()
    return df


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Eikonal Solver")

    # Add the command-line arguments
    parser.add_argument("--config", help="Path to the config file", default="config.ini")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Parse the config file
    config = Config(args.config)

    # Generate velocity model
    if config.method == "checkerboard":
        velocity_model = checkerboard((config.x, config.y), 20) * config.dv + config.v0
    elif config.method == "image":
        velocity_model = read_image(config.image_path, config.dv, config.v0)
    else:
        raise ValueError(f"Invalid method '{config.method}' specified in the config file.")

    # Generate random geometry
    df = random_geometry(config)

    ny, nx = velocity_model.shape
    dx, dy = config.x / nx, config.y / ny

    # Solve Eikonal at source
    eik = tomo_eikonal.Eikonal_Solver(
        velocity_model,
        gridsize=(dy, dx),
        filename=config.fname,
        BL=(11.5, 44.4),
    )
    if config.latlon:
        eik.transform2xy()

    if config.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        image = ax.pcolor(eik.xaxis, eik.zaxis, eik.grid)
        plt.colorbar(image)
        ax.set_aspect(1)
        plt.show()

        plt.pcolor(eik.xaxis, eik.zaxis, eik.grid)
        plt.colorbar()
        plt.scatter(eik.df["xs"], eik.df["ys"], marker="h", c="k")
        plt.scatter(eik.df["xr"], eik.df["yr"], color="r", marker="^")
        plt.xlim([config.xmin, config.xmax])
        plt.ylim([config.ymin, config.ymax])
        plt.show()

    eik.solve()
    eik.calc_traveltimes()
    eik.add_noise(1)
    eik.save_measurements(config.fname)
    print("Done")

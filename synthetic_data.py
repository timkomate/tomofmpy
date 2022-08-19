import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fteikpy import Eikonal2D
import utils.tomo_eikonal as tomo_eikonal
import PIL.Image
import PIL.ImageOps
import pyproj

def checkerboard(shape, tile_size):
    return (np.indices(shape) // tile_size).sum(axis=0) % 2

def read_image(filepath, dv, v0):
    im = PIL.Image.open(filepath)
    gray_image = PIL.ImageOps.grayscale(im)
    velocity_model = ((np.array(gray_image) / 255) * dv) + v0
    velocity_model = np.flipud(velocity_model)
    return velocity_model

def random_geometry(fname, xmin, xmax, ymin, ymax, sigma, nrec, nsrc, latlon = False, plot = False):
    np.random.seed(1234)

    xr = xmin + (xmax - xmin) * np.random.rand(nrec)
    yr = ymin + (ymax - ymin) * np.random.rand(nrec)
    coord_rec = np.vstack([xr, yr]).T

    nsrc = 10
    xs = np.linspace(ymin,ymax,nsrc)
    ys = np.linspace(xmin,xmax,nsrc)
    a = np.tile(ymin,(nsrc))
    b = np.tile(xmax,(nsrc))
    c = np.tile(ymax,(nsrc))
    d = np.tile(xmin,(nsrc))


    coords1 = np.vstack([ys, a])
    coords2 = np.vstack([b, xs])
    coords3 = np.vstack([ys, c])
    coords4 = np.vstack([d, xs])
    coords_source = np.hstack([coords1,coords2, coords3, coords4]).T
    coords_source = (np.unique(coords_source, axis=0))
    print(coords_source.shape)
    n = coords_source.shape[0]

    data = np.zeros((n*r,6))
    c = 0
    for i,source in enumerate(coords_source):
        for j,rec in enumerate(coord_rec):
            data[c] = [i,source[0], source[1], rec[0], rec[1], sigma]
            c += 1
    if latlon:
        df = pd.DataFrame(data, columns = ["source_id", "lons", "lats", "lonr", "latr", "sigma"])
    else:
        df = pd.DataFrame(data, columns = ["source_id", "xs", "ys", "xr", "yr", "sigma"])
    df.astype({'source_id': 'int32'})
    df.to_csv(
        path_or_buf= fname,
        index=False
    )
    if plot:
        plt.scatter(coords1[0], coords1[1])
        plt.scatter(coords2[0], coords2[1])
        plt.scatter(coords3[0], coords3[1])
        plt.scatter(coords4[0], coords4[1])
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.show()
    return df

if __name__ == "__main__":
    y = 120
    x = 120
    xmin, xmax = 1,119
    ymin, ymax = 1,119
    r  = 500
    nsrc = 10
    latlon = False
    plot = False

    dv = 0.1
    v0 = 2

    y = 120
    x = 120

    fname = "synthetic_geometry_xy.csv"
    df = random_geometry(fname, xmin, xmax, ymin,ymax,1, r, nsrc, latlon= False)
    velocity_model = checkerboard((120,120), 20) * dv + v0

    ny,nx = velocity_model.shape
    print(nx,ny)

    print(velocity_model.shape)

    dx, dy = x/nx,y/ny
    print(dx,dy)

    # Solve Eikonal at source
    eik = tomo_eikonal.Eikonal_Solver(velocity_model, gridsize=(dy, dx), filename="./synthetic_geometry_xy.csv", \
                                BL = (11.5,44.4))
    if latlon:
        eik.transform2xy()

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        image = ax.pcolor(eik.xaxis, eik.zaxis,eik.grid)
        plt.colorbar(image)
        ax.set_aspect(1)
        plt.show()

        plt.pcolor(eik.xaxis, eik.zaxis,eik.grid)
        plt.colorbar()
        plt.scatter(eik.df["xs"], eik.df["ys"], marker="h", c = "k")
        plt.scatter(eik.df["xr"], eik.df["yr"], color = 'r', marker="^")
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.show()

    print("Solve")
    eik.solve()
    print("Calc tt")
    eik.calc_traveltimes()
    print("Save")
    eik.save_measurements("./synthetic_measurements_xy.csv")
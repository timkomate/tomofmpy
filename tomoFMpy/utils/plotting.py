import matplotlib.pyplot as plt
import numpy as np


def plot_model_grid(
    xaxis,
    yaxis,
    grid,
    output_path,
    xlabel="X",
    ylabel="Y",
    title=None,
):
    """
    Pseudocolor plot of a 2D grid and save to a PNG.

    Args:
      xaxis: 1D array of x-coordinates.
      yaxis: 1D array of y-coordinates.
      grid: 2D array of shape (len(yaxis), len(xaxis)).
      output_path: Path to save figure (PNG).
      xlabel: Label for X-axis.
      ylabel: Label for Y-axis.
      title: Optional title for the plot.
    """
    fig, ax = plt.subplots()
    mesh = ax.pcolor(xaxis, yaxis, grid, shading="auto")
    fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)


def plot_stations(
    xcoords,
    ycoords,
    output_path,
    marker="o",
    color="k",
    xlabel="X",
    ylabel="Y",
    title=None,
):
    """
    Scatter-plot station coordinates and save to PNG.

    Args:
      xcoords: 1D array of x positions.
      ycoords: 1D array of y positions.
      output_path: Path to save figure (PNG).
      marker: Matplotlib marker style.
      color: Marker color.
      xlabel: Label for X-axis.
      ylabel: Label for Y-axis.
      title: Optional title for the plot.
    """
    fig, ax = plt.subplots()
    ax.scatter(xcoords, ycoords, marker=marker, c=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)


def plot_residual_history(
    residuals,
    output_path,
    xlabel="Iteration",
    ylabel="Misfit",
    title=None,
):
    """
    Plot residual history over iterations and save to PNG.

    Args:
      residuals: Sequence of misfit values.
      output_path: Path to save figure (PNG).
      xlabel: Label for X-axis (iterations).
      ylabel: Label for Y-axis (residual).
      title: Optional title for the plot.
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(residuals)), residuals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_with_stations(
    xaxis,
    yaxis,
    grid,
    src_x,
    src_y,
    rec_x,
    rec_y,
    output_path,
    title=None,
):
    """
    Plot a 2D grid with source and receiver station overlays.

    Args:
      xaxis, yaxis, grid: as in plot_model_grid.
      src_x, src_y: source station coordinates.
      rec_x, rec_y: receiver station coordinates.
      output_path: Path to save figure (PNG).
      title: Optional title.
    """
    fig, ax = plt.subplots()
    mesh = ax.pcolor(xaxis, yaxis, grid, shading="auto")
    fig.colorbar(mesh, ax=ax)
    ax.scatter(src_x, src_y, marker="h", c="white", edgecolors="k", label="Source")
    ax.scatter(rec_x, rec_y, marker="^", c="red", label="Receiver")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_map(
    solver,
    grid,
    output_path,
    title=None,
):
    """
    Plot a 2D grid on a geographic map using Cartopy.

    Args:
      solver: EikonalSolver with transform_to_xy() already called.
      grid: 2D numpy array of same shape as solver grid.
      output_path: File path to save the PNG.
      title: Optional title for the map.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Create mesh in local XY (km)
    X_km, Y_km = np.meshgrid(solver.xaxis, solver.zaxis)
    # Convert to lon/lat
    lon_flat, lat_flat = solver.inv_transformer.transform(
        X_km.ravel() * 1000 + solver.base_xy[0], Y_km.ravel() * 1000 + solver.base_xy[1]
    )
    Lon = lon_flat.reshape(X_km.shape)
    Lat = lat_flat.reshape(Y_km.shape)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Set map extent to data bounds
    ax.set_extent(
        [float(Lon.min()), float(Lon.max()), float(Lat.min()), float(Lat.max())],
        crs=proj,
    )

    # Add natural earth features
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "50m", facecolor="lightgray"
    )
    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", "50m", facecolor="lightblue"
    )
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=2)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"), linestyle="-", linewidth=0.6, zorder=2
    )

    # Plot data
    mesh = ax.pcolormesh(
        Lon, Lat, grid, transform=proj, shading="auto", cmap="seismic_r"
    )

    # Add gridlines with labels
    gl = ax.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        linewidth=0.5,
        linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False

    # Colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.5)
    cbar.set_label("Velocity (km/s)")

    # Titles and labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout(pad=0.0)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

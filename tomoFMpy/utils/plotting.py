import numpy as np
import matplotlib.pyplot as plt


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
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)

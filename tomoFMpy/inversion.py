import argparse
import logging
import os

import numpy as np

from tomoFMpy.core.inversion import Eikonal_Inversion
from tomoFMpy.utils import io, plotting
from tomoFMpy.utils.parameter_init import RealConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run traveltime inversion on real data using Eikonal_Inversion",
    )
    parser.add_argument(
        "--config",
        default="configs/inversion.ini",
        help="Path to the inversion config file",
    )

    args = parser.parse_args()
    logger.info("Using config file: %s", args.config)

    cfg = RealConfig(args.config)
    logger.debug(cfg)

    os.makedirs(cfg.output, exist_ok=True)

    if cfg.use_start_file:
        logger.info("Loading starting model from %s", cfg.start_model)
        start_grid = np.loadtxt(cfg.start_model, delimiter=",")
        if start_grid.shape != (cfg.ny, cfg.nx):
            raise ValueError(
                f"Starting model shape {start_grid.shape} does not match ({cfg.ny}, {cfg.nx})"
            )
    else:
        logger.info("Using constant starting model: %.3f", cfg.const)
        start_grid = np.full((cfg.ny, cfg.nx), cfg.const, dtype=float)

    inv = Eikonal_Inversion(
        modelspace=(cfg.y, cfg.x),
        gridsize=(cfg.ny, cfg.nx),
        measurement_csv=cfg.measurements,
        output_folder=cfg.output,
        bl_corner=(cfg.bl_lon, cfg.bl_lat),
        epsilon=cfg.epsilon,
        eta=cfg.eta,
        use_latlon=cfg.latlon,
    )
    inv.add_starting_model(start_grid.ravel())

    inv.run_linear(method=cfg.method, options={"maxiter": cfg.maxiter})
    plotting.plot_residual_history(
        inv.residuals,
        os.path.join(cfg.output, "residual_history.png"),
    )
    logger.info("Inversion finished. Results in %s", cfg.output)


if __name__ == "__main__":
    main()

import configparser

import numpy as np


class SyntheticConfig:
    def __init__(self, config_file):
        parser = configparser.ConfigParser()
        read_files = parser.read(config_file)
        if not read_files:
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            geom = parser["geometry"]
        except KeyError:
            raise KeyError("Missing required [geometry] section in config")

        self.use_real_stations = geom.getboolean("use_real_stations")

        self.y = geom.getint("y")
        self.x = geom.getint("x")

        self.xmin = geom.getint("xmin")
        self.xmax = geom.getint("xmax")
        self.ymin = geom.getint("ymin")
        self.ymax = geom.getint("ymax")

        self.r = geom.getint("r")
        self.nsrc = geom.getint("nsrc")

        self.latlon = geom.getboolean("latlon")
        self.bl_lon = geom.getfloat("bl_lon")
        self.bl_lat = geom.getfloat("bl_lat")

        self.plot = geom.getboolean("plot")

        try:
            vm = parser["velocity_model"]
        except KeyError:
            raise KeyError("Missing required [velocity_model] section in config")

        self.method = vm.get("method")
        if self.method not in {"checkerboard", "image"}:
            raise ValueError(f"Invalid method '{self.method}' in [velocity_model]")

        self.dv = vm.getfloat("dv")
        self.v0 = vm.getfloat("v0")

        # tile_size is only meaningful if method=checkerboard
        self.tile_size = np.fromstring(
            vm.get("tile_size", fallback=None), dtype=int, sep=","
        )
        if self.method == "checkerboard" and self.tile_size is None:
            raise KeyError("Missing 'tile_size' for checkerboard method")

        # image_path is only meaningful if method=image
        self.image_path = vm.get("image_path", fallback=None)
        if self.method == "image" and not self.image_path:
            raise KeyError("Missing 'image_path' for image method")

        # 3) Parse [general] section
        try:
            gen = parser["general"]
        except KeyError:
            raise KeyError("Missing required [general] section in config")

        self.seed = gen.getint("seed")
        self.fname = gen.get("fname")
        if not self.fname:
            raise KeyError("Missing 'fname' in [general]")

        self.noise = gen.getfloat("noise")

    def __repr__(self):
        return (
            f"Config(geometry: y={self.y}, x={self.x}, xmin={self.xmin}, "
            f"xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax}, r={self.r}, "
            f"nsrc={self.nsrc}, latlon={self.latlon}, bl_lon={self.bl_lon}, "
            f"bl_lat={self.bl_lat}, plot={self.plot}; "
            f"velocity_model: method={self.method}, dv={self.dv}, v0={self.v0}, "
            f"tile_size={self.tile_size}, image_path={self.image_path}; "
            f"general: seed={self.seed}, fname='{self.fname}', noise={self.noise})"
        )


class RealConfig:
    """Configuration for real-data inversion."""

    def __init__(self, config_file: str):
        parser = configparser.ConfigParser()
        read_files = parser.read(config_file)
        if not read_files:
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            inv = parser["inversion"]
            geom = parser["geometry"]
        except KeyError as exc:
            raise KeyError("Missing required section in config") from exc

        self.measurements = inv.get("measurements")
        if not self.measurements:
            raise KeyError("Missing 'measurements' in [inversion]")

        self.output = inv.get("output", fallback="inv_results")
        self.epsilon = inv.getfloat("epsilon", fallback=0.0)
        self.eta = inv.getfloat("eta", fallback=0.0)
        self.use_start_file = inv.getboolean("use_start_file", fallback=False)
        self.start_model = inv.get("start_model", fallback=None)
        self.start_const = inv.getfloat("start_const", fallback=0.0)
        if self.use_start_file and not self.start_model:
            raise KeyError("use_start_file is True but 'start_model' path not provided")
        self.method = inv.get("method", fallback="L-BFGS-B")
        self.maxiter = inv.getint("maxiter", fallback=10)

        self.y = geom.getfloat("y")
        self.x = geom.getfloat("x")
        self.ny = geom.getint("ny")
        self.nx = geom.getint("nx")
        self.latlon = geom.getboolean("latlon", fallback=False)
        self.bl_lon = geom.getfloat("bl_lon", fallback=0.0)
        self.bl_lat = geom.getfloat("bl_lat", fallback=0.0)

    def __repr__(self) -> str:
        return (
            f"RealConfig(inversion: measurements='{self.measurements}', output='{self.output}',"
            f" epsilon={self.epsilon}, eta={self.eta}, use_start_file={self.use_start_file},"
            f" start_model='{self.start_model}', start_const={self.start_const},"
            f" method='{self.method}', maxiter={self.maxiter};"
            f" geometry: y={self.y}, x={self.x}, ny={self.ny}, nx={self.nx},"
            f" latlon={self.latlon}, bl_lon={self.bl_lon}, bl_lat={self.bl_lat})"
        )

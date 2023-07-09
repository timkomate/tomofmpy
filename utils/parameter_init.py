import configparser

class Config:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        geometry_section = self.config["geometry"]
        velocity_model_section = self.config["velocity_model"]
        random_geometry_section = self.config["general"]

        self.y = int(geometry_section.get("y"))
        self.x = int(geometry_section.get("x"))
        self.xmin = int(geometry_section.get("xmin"))
        self.xmax = int(geometry_section.get("xmax"))
        self.ymin = int(geometry_section.get("ymin"))
        self.ymax = int(geometry_section.get("ymax"))
        self.r = int(geometry_section.get("r"))
        self.nsrc = int(geometry_section.get("nsrc"))
        self.latlon = geometry_section.getboolean("latlon")
        self.plot = geometry_section.getboolean("plot")

        self.method = velocity_model_section.get("method")
        self.dv = float(velocity_model_section.get("dv"))
        self.v0 = float(velocity_model_section.get("v0"))

        self.sigma = float(random_geometry_section.get("sigma"))
        self.seed = int(random_geometry_section.get("seed"))
        self.fname = random_geometry_section.get("fname")
        self.image_path = velocity_model_section.get("image_path")
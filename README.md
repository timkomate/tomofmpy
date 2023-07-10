# tomofmpy

Python code for 2D traveltime inversion.

The Eikonal Solver is a program that solves the Eikonal equation for a given velocity model and generates random geometry. It can generate a velocity model using a checkerboard pattern or an image file, and it creates random geometry based on the provided configuration. The solver calculates the travel times and saves the measurements to a file.

## Prerequisites

- Python 3.7 or higher
- Required Python packages: numpy, pandas, matplotlib, Pillow

## Installation

1. Clone the TomoFMpy repository from GitHub:

```shell
git clone https://github.com/timkomate/tomofmpy.git
```
2. Install the required Python packages using pip:

```
pip install -r requirements.txt
```

## Synthetic Data Generation

The synthetic data generation module in MySoftware allows users to generate synthetic seismic data for testing and experimentation. This module includes functionality to generate random geometry and create velocity models using a checkerboard pattern or image files.



### Usage

1. Prepare a configuration file with the desired parameters for synthetic data generation. See the provided `config/synthetic_config.ini` file for reference.

2. Run the `synthetic_data.py` script to generate the synthetic data:

```
python main.py --config config/synthetic_config.ini
```

3. The generated synthetic data will be saved to the specified output files.

### Configuration

The `config.ini` file is used to customize the synthetic data generation process. The following parameters can be adjusted:

- **[geometry]**
  - `y`: Dimensions of the grid along the y-axis.
  - `x`: Dimensions of the grid along the x-axis.
  - `xmin`: Minimum value for x-coordinate.
  - `xmax`: Maximum value for x-coordinate.
  - `ymin`: Minimum value for y-coordinate.
  - `ymax`: Maximum value for y-coordinate.
  - `r`: Number of receiver points.
  - `nsrc`: Number of source points.
  - `latlon`: Whether to use latitude/longitude coordinates (True/False).
  - `plot`: Whether to plot the generated geometry (True/False).

- **[velocity_model]**
  - `method`: Method for generating the velocity model (checkerboard or image).
  - `dv`: Velocity increment.
  - `v0`: Starting velocity.
  - `image_path`: Path to the image file (required if using the image method).

- **[general]**
  - `seed`: Seed for the random number generator.
  - `fname`: Output file name for the generated synthetic dataset.
  - `noise`: Standard deviation gaussian noise added to the synthetic traveltime

### License

This software is licensed under the MIT License.

### Acknowledgments

For any questions or issues, please contact [timko.mate@gmail.com](mailto:timko.mate@gmail.com).


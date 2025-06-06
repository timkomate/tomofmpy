# setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tomoFMpy",
    version="0.1.0",
    author="Mate Timko",
    author_email="timko.mate@gmail.com",
    description="Python package for 2D traveltime inversion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/timkomate/tomofmpy",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pyproj",
        "Pillow",
        "scipy",
        "fteikpy",
    ],
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

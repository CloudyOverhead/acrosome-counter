# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

TORCH_URL = "https://download.pytorch.org/whl"

ver = str(sys.version_info[0]) + str(sys.version_info[1])

requirements = [
    "opencv-python",
    "matplotlib",
    "imgaug",
    "pandas",
]
if os.name != 'nt':
    raise OSError, (
        "Install this repository through `conda env update -n base "
        "--file meta.yaml`."
    )
else:
    print(
        "WARNING: Make sure you have installed CUDA 10.1 and Visual Studio "
        "2019 manually, as well as `cudatoolkit=10.1` through `conda`."
    )
    requirements += [
        (
            f"torch @ {TORCH_URL}"
            f"/cu101/torch-1.6.0%2Bcu101-cp{ver}-cp{ver}m-win_amd64.whl"
        ),
        (
            f"torchvision @ {TORCH_URL}"
            f"/cu101/torchvision-0.7.0%2Bcu101-cp{ver}-cp{ver}m-win_amd64.whl"
        ),
        "detectron2 @ git+https://github.com/DGMaxime/detectron2-windows.git",
    ]

setup(
    name="acrosome-counter",
    version="1.0",
    author="Jérome Simon, Camille Lavoie-Ouellet",
    author_email="jerome.simon@ete.inrs.ca",
    description="Count spermatozoa using deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CloudyOverhead/acrosome-counter",
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=['setuptools-git'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

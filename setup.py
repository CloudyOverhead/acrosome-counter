# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "opencv-python",
    "matplotlib",
    "imgaug",
    "pandas",
]
if os.name != 'nt':
    print("WARNING: Make sure to install `cudatoolkit=10.2` through `conda`.")
    requirements += [
        "ninja",
        "pytorch==1.8.0",
        "torchvision==0.9.0",
        "git+https://github.com/facebookresearch/detectron2.git@v0.3",
    ]
else:
    print(
        "WARNING: Make sure to install `cudatoolkit=10.1` through `conda`."
    )
    requirements += [
        "pytorch==1.6.0",
        "torchvision==0.7.0",
        "git+https://github.com/DGMaxime/detectron2-windows.git",
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

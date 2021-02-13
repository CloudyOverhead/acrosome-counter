# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from subprocess import run
from os import chdir
from os.path import join, pardir, abspath, exists
from shutil import copyfile


with open("README.md", "r") as fh:
    long_description = fh.read()

MODELS_PATH = join(abspath('.'), 'models', 'research')
if not exists(MODELS_PATH):
    run(["git", "clone", "https://github.com/tensorflow/models.git"])
chdir(MODELS_PATH)
run(
    ["protoc", join("object_detection", "protos", "*.proto"), "--python_out=."]
)
copyfile(join("object_detection", "packages", "tf2", "setup.py"), "setup.py")
chdir(join(pardir, pardir))

REQUIREMENTS = [
    "tensorflow-gpu",
    "matplotlib",
    f"object_detection @ file://localhost/{MODELS_PATH}",
]

setup(
    name="acrosome-counter",
    version="0.0.0",
    author="Jérome Simon, Camille Lavoie-Ouellet",
    author_email="jerome.simon@ete.inrs.ca",
    description="Count spermatozoa using deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CloudyOverhead/acrosome-counter",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    setup_requires=['setuptools-git'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

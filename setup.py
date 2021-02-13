from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIREMENTS = [
    "tensorflow-gpu",
    "matplotlib",
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

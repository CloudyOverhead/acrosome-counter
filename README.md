# acrosome-counter: Count spermatozoa using deep learning

![](https://raw.githubusercontent.com/CloudyOverhead/acrosome-counter/main/example.png)


## Install instructions

- Install [Anaconda](https://www.anaconda.com/products/individual) (check *Add Anaconda3 to my PATH environment variable* upon installation)
- [Download this project](https://github.com/CloudyOverhead/acrosome-counter/releases)
- Go in the directory where it was downloaded, unpack the project and go into the project's directory
- On command line interface, navigate to the project's directory
- On MacOS or Linux:
  - Run `conda env update -n base --file meta.yaml`
- On Windows:
  - Make sure to [have a Nvidia GPU](https://www.windowscentral.com/how-determine-graphics-card-windows-10)
  - Install [Git](https://git-scm.com/downloads)
  - Install [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
  - Install [Visual Studio 2019](https://visualstudio.microsoft.com/fr/downloads/) (install all C++ build tools)
  - Restart your computer
  - Run `conda install cudatoolkit=10.1`
  - Run `pip install -e .` (within the project's directory)
- Download the [trained model](https://drive.google.com/file/d/1loadwjn-4cIuj-E_2SJrmGA3MV1xxrE0/view?usp=sharing)
- Put the trained model under a new subdirectory called `logs`


## Usage

At inference time, navigate to the directory you want to run predictions on. Then, from command line interface, specify either:
- `python -m acrosome_counter . --infer` to produce predictions. You may also add `--plot` to plot the results during execution.
- `python -m acrosome_counter . --review` to review the predictions produced previously. You may also use `python -m acrosome_counter . --infer --review` to automatically fall into review mode after inference.

`acrosome-counter` saves predictions as a XML file under the current directory (`.`) and statistics as a CSV file. The XML file is compatible with [CVAT](https://github.com/openvinotoolkit/cvat). The CSV file can be opened from any software that can parse tabular data, such as Microsoft Excel.


## Issues

Ask for help directly in [GitHub's Issues tab](https://github.com/CloudyOverhead/acrosome-counter/issues).


## Further inquiries

If you have specific needs for deep learning solutions, contact me at <jerome@geolearn.ai> or <info@geolearn.ai>. Geolearn provides automated machine learning solutions for geosciences, but also general purpose artificial intelligence tools.

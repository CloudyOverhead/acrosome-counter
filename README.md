# acrosome-counter

Count spermatozoa using deep learning.

![](https://raw.githubusercontent.com/CloudyOverhead/acrosome-counter/main/example.png)


## Install instructions

- Install [Anaconda](https://www.anaconda.com/products/individual)
- [Download this project](https://github.com/CloudyOverhead/acrosome-counter/releases)
- Go in the folder where it was downloaded
- On command line interface, type `conda env update -n base --file meta.yaml`
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

If you have specific needs for deep learning solutions, contact me at <jerome@geolearn.ai> or <info@geolearn.ai>.

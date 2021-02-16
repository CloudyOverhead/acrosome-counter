# acrosome-counter
Count spermatozoa using deep learning.

## Install instructions

- Install [Anaconda](https://www.anaconda.com/products/individual)
- Install [Git](https://git-scm.com/downloads)
- Clone this repository locally, that is, on command line interface (search and `cmd` in Windows menu), type `git clone https://github.com/CloudyOverhead/acrosome-counter.git`
- Go in the folder where it was cloned
- Download [`protobuf`](https://github.com/protocolbuffers/protobuf/releases/)
- Unpack and copy `bin/protoc.exe` in `acrosome-counter/models/research`
- On command line interface, type `pip install .`

If your computer has a [NVIDIA graphics card](https://nvidia.custhelp.com/app/answers/detail/a_id/2040/~/identifying-the-graphics-card-model-and-device-id-in-a-pc), you may [install the prerequisites for TensorFlow GPU 2.14.1](https://www.tensorflow.org/install/gpu?hl=fr).

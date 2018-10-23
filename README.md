# pyromancy
## PyTorch Black Magic

## Installation

You should be using python 3. To create a python 3 env with conda:

`conda create --name pytorch python=3 numpy`

### Steps for installation:

1. `pip -r requirements.txt`
2. `conda install pytorch torchvision cuda80 -c soumith`
3. `python setup.py develop`

The develop is because this allows for python to point to the library rather
than a once-compiled egg. In this way, Python can detect any file changes.
# QAP Project

Approximate solution algorithms for QAP 

All data was downloaded from [QAPLIB](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/#KP)
I haven't found any license file, so I guess this note is enough

## Installation

use this command to install library in editable mode:

```bash
$ git clone https://github.com/Zhylkaaa/qap-project.git
$ cd qap-project
$ python -m pip install -e .
```


## Development
One can also setup flake8, flake8-docstrings and YAPF to follow codestype.

To run checks use:
```bash
yapf -i --style google --recursive .
flake8 --docstring-convention google --max-line-length 119 --ignore=W504,E121,E704,E123,E226,E126,W503,E24,D100,D104 .
```

This will be added to setup.cfg in future 

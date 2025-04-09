# eosimutils
Common utilities for packages and scripts within the EarthObservationSimulator organization

**Currently under active development.**

## Installation

Requires: Unix-like operating system, `python 3.13`, `pip`

Create a conda environment:
```
conda create -n eosim-revised python=3.13
conda activate eosim-revised
conda install sphinx
pip install sphinx-rtd-theme
pip install pylint
pip install black
pip install coverage
pip install skyfield
pip install astropy
pip install scipy

make install
```
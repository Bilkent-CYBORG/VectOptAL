## Vector Optimization with Active Learning

[![Test Workflow](https://github.com/Bilkent-CYBORG/VectOptAL/actions/workflows/test.yml/badge.svg)](https://github.com/Bilkent-CYBORG/VectOptAL/blob/master/.github/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/vectoptal/badge/?version=latest)](https://vectoptal.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### Examples and Documentation

Visit our [**website**](https://vectoptal.readthedocs.io/en/latest/) to see tutorials, examples and API references on how to use VectOptAL.


### Setup

For requirements, see [requirements.txt](requirements.txt) or [environment.yml](environment.yml).

Installation using pip:
```bash
pip install vectoptal
```

<!-- To setup a proper environment:
```setup
conda env create --name vo --file environment.yml
``` -->

#### Latest (Unstable) Version
To upgrade to the latest (unstable) version, run

```bash
pip install --upgrade git+https://github.com/Bilkent-CYBORG/VectOptAL.git
```

#### Manual installation (for development)

If you are contributing a pull request, it is best to perform a manual installation:

```sh
git clone https://github.com/Bilkent-CYBORG/VectOptAL.git
cd VectOptAL
pip install -e .[dev,docs,examples,test]
```

<!-- 
### Run the example experiment as:
```bash
python main.py
``` -->

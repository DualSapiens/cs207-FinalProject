[![Build Status](https://travis-ci.org/DualSapiens/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/DualSapiens/cs207-FinalProject.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/DualSapiens/cs207-FinalProject/badge.svg?branch=master)](https://coveralls.io/github/DualSapiens/cs207-FinalProject?branch=master)

## Gradpy: A tool for automatic differentiation
### cs207-FinalProject

**Group Name:** DualSapiens

**Group Number:** 13

**Members:** Jovana Andrejevic, Gopal Kotecha, Jay Li, Ziyi Zhou

This is the main repository for the `gradpy` package, which includes an `autodiff` module for automatic differentiation, and a `math` module for compatibility with special functions. This repository also hosts the `therapy_planner` package, an application of automatic differentiation for dose delivery optimization relevant to Intensity Modulated Radiation Therapy (IMRT), that automatically ships with `gradpy`.
The **latest [documentation]** for `gradpy` and the featured application `therapy_planner` is hosted on **readthedocs**.

## Quick `gradpy` installation guide

1. We suggest working with our packages within a virtual environment. To do so, ensure that `virtualenv` for Python 3 has been installed.

2. Create a new virtual environment `env`:
```
virtualenv env --python=python3
```

3. Activate the environment:
```
source env/bin/activate
```

4. Install `gradpy`:
```
pip install gradpy
```

5. Users can now try the examples shown in the *Usage* section of the **[documentation]** to get started!

## Testing `gradpy`

After installation, users may wish to run tests to validate their installed package is working properly. `gradpy` comes with a test suite that may be easily run using `pytest`.

1. Within the virtual environment in which `gradpy` has been installed, install pytest:
```
pip install pytest
```
**note:** A terminal restart after installing `pytest` is likely necessary for changes to take effect.

2. Run the `gradpy` test suite:
```
pytest --pyargs gradpy
```

## Quick `therapy_planner` installation guide

`therapy_planner` may be installed in the same manner as `gradpy`.

1. Within a virtual environment, install `therapy_planner`:
```
pip install therapy_planner
```

2. Users can now follow the demos in the *Featured Application* section of the **[documentation]**.

## Testing `therapy_planner`

1. Ensure that `pytest` is installed in the virtual environment.

2. Run the `therapy_planner` test suite:
```
pytest --pyargs therapy_planner
```

[documentation]: https://autodiff.readthedocs.io/en/latest

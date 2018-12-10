Installation
================
Install Gradpy Through PyPI
---------------------------
Because we want the users to be able to use our package right away, we suggest the users to install our package in a virtual environment. Here are some suggested steps to install our package for the users:

1. Ensure that `virtualenv` for `python3` has been installed.
2. Type `virtualenv env --python=python3` to create a new virtual environment.
3. Type `source env/bin/activate`. You have now activated a virtual environment.
4. Type `pip install gradpy` to install the `gradpy` package.
5. Users can now try the examples shown in :doc:`Usage` page or `docs/examples`_ in our GitHub repo to verify and explore the wonderful `gradpy` package!

Install Manually
----------------------
If the users want to install our package manually, here are the few steps we suggest them to do:

1. Download `gradpy-0.0.1.tar.gz`_ and `requirements.txt`_ from our Github repository.
2. Create a project directory and `cd` into that directory.
3. Ensure that `virtualenv` for `python3` has been installed.
4. Type `virtualenv env --python=python3` to create a new virtual environment.
5. Type `source env/bin/activate`. You have now activated a vitrual environment.
6. Place the downloaded `requirements.txt` file into the project directory and type `pip install -r requirements.txt` to install the necessary dependencies.
7. Unpack `gradpy-0.0.1.tar.gz` in the project directory (or anywhere really), and `cd` into the unpacked directory.
8. Type `python setup.py install` and the package will be installed.
9. Users can now try the examples shown in :doc:`Usage` page or `docs/examples`_ in our GitHub repo to verify and explore the wonderful `autodiff` package!

Testing
---------
After installation, users may wish to run tests to validate their installed package is working properly. `gradpy` comes with a test suite that may be easily run using `pytest`.

1. Within the virtual environment in which `gradpy` has been installed, run `pip install pytest``.
**note:** A terminal restart after installing `pytest` is likely necessary for changes to take effect.
2. Type `pytest --pyargs gradpy` to run the `gradpy` test suite.

.. _gradpy-0.0.1.tar.gz: https://github.com/DualSapiens/cs207-FinalProject/blob/master/gradpy/dist/gradpy-0.0.1.tar.gz

.. _requirements.txt: https://github.com/DualSapiens/cs207-FinalProject/blob/master/gradpy/requirements.txt

.. _docs/examples: https://github.com/DualSapiens/cs207-FinalProject/tree/master/docs/examples
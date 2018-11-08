Installation
================
Install Through Test PyPI
---------------------------
Because we want the users to be able to use our package right away, we suggest the users to install our package in a virtual environment. Here are some suggested steps to install our package for the users:

1. Type `sudo easy_install virtualenv` in the command line to check whether `virtualenv` has been installed.
2. Type `virtualenv env --python=python3` to create a new virtual environment.
3. Type `source env/bin/activate`. You have now activated a vitrual environment.
4. Type `python3 -m pip install --index-url https://test.pypi.org/simple/ autodiff` to install the `autodiff` package.
5. Now try running python and import the `autodiff` package.

Install Manually
----------------------
If the users want to install our package manually, here are the few steps we suggest them to do:

1. Download `autodiff-0.0.6.tar.gz`_ and `requirements.txt`_ from our Github repository.
2. Create a project directory and `cd` into that directory.
3. Ensure that `virtualenv` for `python3` has been installed.
4. Type `virtualenv env --python=python3` to create a new virtual environment.
5. Type `source env/bin/activate`. You have now activated a vitrual environment.
6. Place the downloaded `requirements.txt` file into the project directory and type `pip install -r requirements.txt` to install the necessary dependencies.
7. Unpack `autodiff-0.0.6.tar.gz` in the project directory (or anywhere really), and `cd` into the unpacked directory.
8. Type `python setup.py install` and the package will be installed.
9. You can now try the example usages shown in `docs/examples`_ on our GitHub repo to verify and explore the wonderful `autodiff` package!

.. _autodiff-0.0.6.tar.gz: https://github.com/DualSapiens/cs207-FinalProject/blob/master/autodiff/dist/autodiff-0.0.6.tar.gz

.. _requirements.txt: https://raw.githubusercontent.com/DualSapiens/cs207-FinalProject/master/autodiff/requirements.txt

.. _docs/examples: https://github.com/DualSapiens/cs207-FinalProject/tree/master/docs/examples
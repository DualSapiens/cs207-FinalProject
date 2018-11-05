Documentation of AutoDiff Package
====================================

.. toctree::
   :maxdepth: 3

   Background
   Software Organization
   Implementation


Introduction
--------------
The *autodiff* package implements `automatic differentiation`_, a technique
for computing
derivatives of functions that is distinct from both symbolic and numerical differentiation.

The primary advantage of automatic differentiation is its ability to handle
complicated functions without sacrificing the accuracy of the computed
derivative. Whereas symbolic differentiation guarantees accuracy, it can be
intractable for complex functions. By contrast, numerical differentiation
affords ease in implementation and is amenable to any function, but it comes at the expense of accuracy. Automatic, or algorithmic, differentation is an alternative approach which allows derivatives to be computed by a computer program up to **machine precision**.

Automatic differentiation in science and engineering has been applied to address problems in a variety of areas, including optimization, root-finding, and implicit time-integration.

.. _automatic differentiation: https://en.wikipedia.org/wiki/Automatic_differentiation

How to Use AutoDiff
-------------------
Install Through Test PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^
Because we want the users to be able to use our package right away, we suggest the users to install our package in a virtual environment. Here are some suggested steps to install our package for the users:

1. Type `sudo easy_install virtualenv` in the command line to check whether `virtualenv` has been installed.
2. Type `virtualenv env`.
3. Type `source env/bin/activate`. You now created and activated a vitrual environment.
4. Type `pip install -i https://test.pypi.org/simple/ autodiff`
5. Now try running python and import the `autodiff` package.

Install Manually
^^^^^^^^^^^^^^^^^^
If the users want to install our package manually, here are the few steps we suggest them to do:

1. Download the file `cs207-FinalProject/autodiff/dist/autodiff-0.0.1.tar.gz` and `cs207-FinalProject/requirements.txt` from our Githun repository.
2. Create a directory called `myproj` and unpack the `autodiff-0.0.1.tar.gz` file into that directory.
3. Type `sudo easy_install virtualenv` in the command line to check whether `virtualenv` has been installed.
4. Type `virtualenv env`.
5. Type `source env/bin/activate`. You now created and activated a vitrual environment.
6. Type `pip install -r requirements.txt`
7. Under the directory, run python and import autodiff.


.. Below are comments - just hiding for now
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

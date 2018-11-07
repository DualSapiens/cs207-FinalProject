Installation
================
Install Through Test PyPI
---------------------------
Because we want the users to be able to use our package right away, we suggest the users to install our package in a virtual environment. Here are some suggested steps to install our package for the users:

1. Type `sudo easy_install virtualenv` in the command line to check whether `virtualenv` has been installed.
2. Type `virtualenv env` to create a new virtual environment.
3. Type `source env/bin/activate`. You have now activated a vitrual environment.
4. Type `python3 -m pip install --index-url https://test.pypi.org/simple/ autodiff` to install the `autodiff` package.
5. Now try running python and import the `autodiff` package.

Install Manually
----------------------
If the users want to install our package manually, here are the few steps we suggest them to do:

1. Download the file `cs207-FinalProject/autodiff/dist/autodiff-0.0.6.tar.gz` and `cs207-FinalProject/requirements.txt` from our Github repository.
2. Create a project directory called `myproj` and unpack the `autodiff-0.0.6.tar.gz` file into that directory.
3. Type `sudo easy_install virtualenv` in the command line to check whether `virtualenv` has been installed.
4. Type `virtualenv env` to create a new virtual environment.
5. Type `source env/bin/activate`. You have now activated a vitrual environment.
6. Type `pip install -r requirements.txt` to install the necessary dependencies.
7. Under the project directory, run python and import autodiff.





Software Organization
======================
Directory Structure
---------------------
.. code-block:: python

    autodiff\
        autodiff\
            tests\
                __init__.py
                test_autodiff.py
            __init__.py
            autodiff.py
            math.py
        setup.py
        LICENSE
        README.md
    docs\
        milestone1.ipynb
        examples\
            intro_AD.py
            multivariate_AD.py
            vector_valued_AD.py
            special_functions.py
    README.md
    .travis.yml
    setup.cfg
    requirements.txt

Modules
-------------
1. **autodiff/autodiff.py**: defines the core structure of automatic differentiation, including   attributes for accessing values and gradients, and methods for operator overloading. This is the main module users will import.

2. **autodiff/math.py**: defines elementary functions (sin, exp) to be imported by the user and used when defining a function.

4. **docs/examples/intro_AD.py**: An introductory example of using the AutoDiff package, covering univariate, scalar valued functions.
5. **examples/multivariate_AD.py**: A demonstration of automatic differentiation for multivariate functions, as well as retrieving the attributes of vector-valued functions.
6. **examples/special_functions.py**: How to import and use functions such as `sin` and `exp` when defining functions.
7. **autodiff/tests/test_autodiff.py**: A module of test functions that we will use in validating our implementation.

Tests
-------
The test suite is in the *tests* directory of our package. We maintain our test suite and package on GitHub, with Travis CI and Coveralls for continuous integration. We use PyPI for package distribution.
Software Organization
======================
Directory Structure
---------------------
.. code-block:: python

    gradpy\
        gradpy\
            tests\
                __init__.py
                test_autodiff.py
            __init__.py
            autodiff.py
            math.py
        setup.py
        requirements.txt
        LICENSE
        README.md
    docs\
        milestone1.ipynb
        examples\
            introductory_demo.ipynb
            special_functions_demo.ipynb
            newton_multivar_demo.ipynb
            newton_multivar.py
    README.md
    .travis.yml
    setup.cfg

Modules
-------------
1. **gradpy/autodiff.py**: defines the core structure of automatic differentiation, including   attributes for accessing values and gradients, and methods for operator overloading. This is the main module users will import.
2. **gradpy/math.py**: defines elementary functions (sin, exp) to be imported by the user and used when defining a function.
3. **gradpy/tests/test_autodiff.py**: A module of test functions used in validating our implementation.

Examples
-------------
1. **docs/examples/introductory_demo.ipynb**: Introductory examples of using the ``gradpy`` package, covering single-variable, multi-variable, and vector-valued functions.
2. **docs/examples/special_functions_demo.ipynb**: How to import and use functions such as ``sin`` and ``exp``.
3. **docs/examples/newton_multivar_demo.ipynb**: An application of ``gradpy`` in root-finding using Newton's method.


Tests
-------
Our test suite is maintained along with our package on GitHub, with Travis CI and Coveralls for continuous integration. Instructions for running the test suite after successfully installing ``gradpy`` can be found in the :doc:`Installation` section.
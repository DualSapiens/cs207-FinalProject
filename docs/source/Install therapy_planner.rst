Installation and Software Organization
========================================

Installation
^^^^^^^^^^^^^

``therapy_planner`` can be easily installed with ``pip`` and is suitable for an environment running Python 3. We suggest working with a virtual environment when using our package.

1. Ensure that ``virtualenv`` for ``python3`` has been installed.
2. Create a new virtual environment: ``virtualenv env --python=python3``
3. Activate the environment: ``source env/bin/activate``.
4. Install ``therapy_planner``: ``pip install therapy_planner``.

Testing
^^^^^^^^^

``pytest`` may be used to run the test suite that comes with ``therapy_planner``.

1. Within the virtual environment, install ``pytest``: ``pip install pytest``. **note**: A terminal restart is likely necessary after installing ``pytest`` for changes to take effect.
2. Run the ``therapy_planner`` test suite: ``pytest --pyargs therapy_planner``.

Software Organization
^^^^^^^^^^^^^^^^^^^^^^^

``therapy_planner`` is maintained on GitHub_, with Travis CI and Coveralls for continuous integration. 

Directory Structure
---------------------
.. code-block:: python

    therapy_planner\
        therapy_planner\
            tests\
                __init__.py
                test_interface.py
                test_costfunctions.py
                test_bfgs.py
            __init__.py
            interface.py
            costfunctions.py
            bfgs.py
        setup.py
        requirements.txt
        LICENSE
        README.md
        MANIFEST.in
    docs\
        examples\
            bfgs_demo.ipynb
            optimize_demo.ipynb
    README.md
    .travis.yml
    setup.cfg

Modules
-------------
1. **therapy_planner/interface.py**: This is the main module users will import, which defines a user-friendly ``PlannerInterface`` Class to read in input data, run optimization, and visualize the results.
2. **therapy_planner/costfunctions.py**: Defines different contributions to the cost function used in optimization, including mean squared error and a logistic penalty function.
3. **therapy_planner/bfgs.py**: Includes the BFGS routine implemented with ``gradpy``, which is used by ``therapy_planner`` to perform a minimization of the cost function.
4. **gradpy/tests/**: A directory of test modules used in validating our implementation.

Examples
-------------
1. **docs/examples/bfgs_demo.ipynb**: An example of objective function minimization with ``gradpy`` using the BFGS algorithm.
2. **docs/examples/optimize_demo.ipynb**: Several demos of use cases for ``therapy_planner``, including handling missing data, adding and tuning penalties to the cost function, and making use of different built-in visualizations of the results.

.. _GitHub: https://github.com/DualSapiens/cs207-FinalProject
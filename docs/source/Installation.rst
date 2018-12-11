Installation
================
Install ``gradpy`` Through PyPI
---------------------------
Because we want the users to be able to use our package right away, we suggest the users to install our package in a virtual environment. Here are some suggested steps to install our package for the users:

1. Ensure that ``virtualenv`` for ``python3`` has been installed.
2. Type ``virtualenv env --python=python3`` to create a new virtual environment.
3. Type ``source env/bin/activate``. You have now activated a virtual environment.
4. Type ``pip install gradpy`` to install the ``gradpy`` package.
5. Users can now try the examples shown in :doc:`Usage` page or `docs/examples`_ in our GitHub repo to verify and explore the wonderful ``gradpy`` package!

Testing ``gradpy``
---------
After installation, users may wish to run tests to validate their installed package is working properly. ``gradpy`` comes with a test suite that may be easily run using ``pytest``.

1. Within the virtual environment in which ``gradpy`` has been installed, run ``pip install pytest``. **note:** A terminal restart after installing ``pytest`` is likely necessary for changes to take effect.
2. Type ``pytest --pyargs gradpy`` to run the ``gradpy`` test suite.



.. _docs/examples: https://github.com/DualSapiens/cs207-FinalProject/tree/master/docs/examples
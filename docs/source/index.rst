Gradpy and Therapy Planner Documentation
====================================

.. toctree::
   :maxdepth: 2

   Overview <self>
   Installation
   Usage
   Background
   Software Organization
   Implementation
   Featured Application
   Summary and Future

Introduction
--------------
The ``gradpy`` package, through its primary module ``autodiff``, implements `automatic differentiation`_, a technique for computing derivatives of functions that is distinct from both symbolic and numerical differentiation.

The primary advantage of automatic differentiation is its ability to handle
complicated functions without sacrificing the accuracy of the computed
derivative. Whereas symbolic differentiation guarantees accuracy, it can be
intractable for complex functions. By contrast, numerical differentiation
affords ease in implementation and is amenable to any function, but it comes at the expense of accuracy. Automatic, or algorithmic, differentation is an alternative approach which allows derivatives to be computed by a computer program up to **machine precision**.

Automatic differentiation in science and engineering has been applied to address problems in a variety of areas, including optimization, root-finding, and implicit time-integration.

The ``therapy_planner`` package is an application of automatic differentiation. See more about ``therapy_planner`` in :doc:`Featured Application` page.

This project also hosts the therapy_planner package, an application of automatic differentiation for dose delivery optimization relevant to Intensity Modulated Radiation Therapy (IMRT), that automatically ships with gradpy.

This project_ is hosted on GitHub.

.. _automatic differentiation: https://en.wikipedia.org/wiki/Automatic_differentiation

.. _project: https://github.com/DualSapiens/cs207-FinalProject




.. Below are comments - just hiding for now
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

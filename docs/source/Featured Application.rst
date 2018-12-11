Featured Application: Intensity Modulated Radiation Therapy (IMRT)
===================================================================

One of the most prominent applications of automatic differentiation has been in the topic of optimization. Optimization problems typically entail minimizing a cost or objective function by following the direction of greatest decrease, provided by its gradient. In particular, optimization in the field of Intensity Modulated Radiation Therapy (IMRT) relies on precise gradients of complex, multivariate cost functions subject to constraints, and is thus a well-suited problem for automatic differentiation. Inspired by the work of Jee *et al.* [JeMF18]_, we have developed the ``therapy_planner`` package to solve 2D dose delivery optimization problems in IMRT. ``therapy_planner`` ships with ``gradpy`` and comes with a user-friendly interface for data input, optimization, and visualization. Though an introductory package, ``therapy_planner`` flexibly handles missing data, allows users to adjust the stringency of optimization criteria, and optionally optimizes over different orientations of the target region, to name a few of the features we hope users will explore with this application.

The sections below outline some of the basic concepts of IMRT, detail the optimization routine adapted for ``gradpy`` that is at the core of ``therapy_planner``, and present several example demos to get started.

.. toctree::
   :maxdepth: 2

   Overview <self>
   Basics of IMRT
   bfgs_demo
   optimize_demo
   Install therapy_planner <Install therapy_planner>

.. rubric:: References

.. [JeMF18] Jee, Kyung-Wook, Daniel L. McShan, and Benedick A. Fraass. "Implementation of Automatic Differentiation Tools for Multicriteria IMRT Optimization." *Automatic Differentiation: Applications, Theory, and Implementations*, Springer, 2005.
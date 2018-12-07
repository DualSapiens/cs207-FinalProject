Summary and Future
===================

So far, we have developed the `gradpy` package, which enables users to calculate exact derivatives and Jacobian
matrix with respect to their input functions and variables. We cover the needs of calculating the derivatives on the most
common types of functions.

We have also built a suite to solve a simple (2-Dimensional) version of the Intensity Modulated Radiation Therapy
(IMRT) optimization problem. In order to solve this type of problem, we construct the Broyden–Fletcher–Goldfarb–Shanno
(BFGS) algorithm rooted in our `gradpy` package.  We have designed this in an easy to use manner, so that clinical (non engineer) users can use it in an accessible fashion.  

There are a number of future directions that this work could take.  The first would be to extend the system to three dimensions.  It would also be valuable to serve different types of radiotherapies, and extend the package into a comprehensive radiotherapy optimisation solution.  
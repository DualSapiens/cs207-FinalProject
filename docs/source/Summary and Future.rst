Summary and Future
========

So far, we have developed the `gradpy` package, which enables the users to calculate the exact derivatives and Jacobian
matrix with respect to their input functions and variables. We cover the needs of calculating the derivatives on the most
common types of functions.

We have also built a program to solve a simple (2-Dimensional) version of the Intensity Modulated Radiation Therapy
(IMRT) optimization problem. In order to solve this type of problem, we construct the Broyden–Fletcher–Goldfarb–Shanno
(BFGS) algorithm rooted in our `gradpy` package. In addition, we build a nice interface so that any users
in that field who do not have many experiences with coding can quickly
solve their 2-D IMRT problems using our program.

One possible improvement on our IMRT solver is to enable it to solve problems in higher dimensions.
We are also looking for more cool applications that could be built up from our `gradpy` package in addition to IMRT.

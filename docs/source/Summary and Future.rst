Summary and Future
===================

Thus in summary, we have developed the ``gradpy`` package, which enables users to calculate exact derivatives and Jacobian
matrix with respect to their input functions and variables. We cover the needs of calculating the derivatives on the most
common types of functions.

To apply this package, we have built a suite to solve a (2-Dimensional) version of the Intensity Modulated Radiation Therapy (IMRT) optimization problem. This 2 dimensional version would immediately be applicable to 2 dimensional radiotherapy cases such as skin cancers, and could serve as the foundation for an extended package that would work in three dimensions.  In order to solve this type of problem, we implement the Broyden–Fletcher–Goldfarb–Shanno
(BFGS) algorithm rooted in our ``gradpy`` package.  We have designed this in an easy to use manner, so that clinical (non engineer) users can use it in an accessible fashion.  

There are a number of future directions that this work could take.  The first would be to extend the system to three dimensions.  It would also be valuable to serve different types of radiotherapies.  IMRT is currently being used in a variety of cancers: prostate, head & neck, lung, brain, gastrointestinal and breast.  It is used specifically because these cancers are generally located close to critical organs.  It could therefore be valuable to make domain specific versions of the package, extending it into a comprehensive radiotherapy optimisation solution.  
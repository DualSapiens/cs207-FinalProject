import numpy as np

def BFGS(fun, var, xi, tol=1.0e-8, maxiter=100000):
    """ Broyden-Fletcher-Goldfarb-Shanno algorithm for unconstrained nonlinear optimization,
        using automatic differentiation.

    INPUTS
    =======
    fun: an autodiff Operation object (scalar-valued function) whose value to minimize.
    var: a list of autodiff Var objects that are independent variables of fun.
         The values of var are updated to the optimal solution on return.
    xi: a list of initial guesses for each variable in var.
    tol (optional, default 1e-8): the stopping tolerance for the 2-norm of the step size.
    maxiter (optional, default 100000): the maximum number of iterations.

    RETURNS
    =======
    step: the 2-norm of the final step size.
    Niter: the number of iterations performed.
    found: whether the minimum has been found.
    """

    found = False  # A variable to indicate whether the minimum to the cost function is found
    try:
        len(fun)
        raise Exception("Function must be scalar-valued.")
    except TypeError:
        try:
            len(var)  # multivariate function
        except TypeError:  # univariate function
            var = [var]
            xi = [xi]
        for v, val in zip(var, xi):
            v.set_value(val)
        s = 1.0e20
        Hinv = np.identity(len(var))  # approx inverse Hessian
        Niter = 0
        while True:
            if np.linalg.norm(s) <= tol:
                found = True  # Set to True because minimum has been found
                print("Minimum found.")
                break
            elif Niter == maxiter:
                print("Reached maximum number of iterations.")
                break
            else:
                g = fun.grad(var)
                s = -Hinv.dot(g.T)
                y = -g
                for v, step in zip(var, s):
                    v.set_value(v.value+step)
                y += fun.grad(var)
                rho = 1.0/np.dot(y, s)
                Hinv = np.dot((np.identity(len(var))-rho*np.outer(s, y)),\
                    np.dot(Hinv,(np.identity(len(var))-rho*np.outer(y, s)))) + rho*np.outer(s, s)
                Niter += 1
        return np.linalg.norm(s), Niter, found

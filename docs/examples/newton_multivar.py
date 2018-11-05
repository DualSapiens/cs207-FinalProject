
import numpy as np

def newton_multivar(fun,var,xi,tol=1.0e-12,maxiter=100000):
    """ Multivariate Newton root-finding using automatic differentiation.

    INPUTS
    =======
    fun: an autodiff Operation or Array object whose roots are to be computed.
    var: an array of autodiff Var objects that are independent variables of fun.
    xi: an array of initial guesses for each variable in var.
    tol (optional, default 1e-12): the stopping tolerance for the 2-norm of residual.
    maxiter (optional, default 100000): the maximum number of iterations.

    RETURNS
    =======
    root: the root to which the method converged.
    res: the 2-norm of the residual.
    Niter: the number of iterations performed.
    traj: the trajectory taken by each variable through successive iterations. Each
          column of traj corresponds to an independent variable in var.
    """

    h = np.ones(len(var)) # initial step size
    for v,val in zip(var,xi):
        v.set_value(val)
    traj = np.array(xi)
    b = fun.value
    Niter = 0
    while (np.sqrt(sum([v**2 for v in fun.value]))>tol and Niter<maxiter):
        J = fun.grad(var)
        h = -np.linalg.solve(J,b)
        for v,step in zip(var,h):
            v.set_value(v.value+step)
        traj = np.vstack([traj,[v.value for v in var]])
        b = fun.value
        Niter += 1
    root = [v.value for v in var]
    res = np.sqrt(sum([v**2 for v in fun.value]))
    return root,res,Niter,traj
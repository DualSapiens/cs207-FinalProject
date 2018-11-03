import sys
sys.path.append("../../autodiff/autodiff")
import numpy as np
from autodiff import Var, Array

def newton_multidim(fun,var,xi,tol=1.0e-12,maxiter=100000):
    """ Multidimensional Newton root-finding using automatic differentiation.

    INPUTS
    =======
    fun: an autodiff Operation object whose roots are to be computed.
    var: an array of autodiff Var objects that are independent variables of fun.
    xi: an array of initial guesses for each variable in var.
    tol (optional, default 1e-12): the stopping tolerance for the 2-norm of residual.
    maxiter (optional, default 100000): the maximum number of iterations.

    RETURNS
    =======
    root: the root to which the method converged.
    res: the 2-norm of the residual.
    Niter: the number of iterations performed.
    """

    h = np.ones(len(var)) # initial step size
    for v,val in zip(var,xi):
        v.set_value(val)
    b = np.array(fun.value)
    Niter = 0
    while (np.sqrt(sum([v**2 for v in fun.value]))>tol and Niter<maxiter):
        J = f.grad(var)
        h = -np.linalg.solve(J,b)
        for v,step in zip(var,h):
            v.set_value(v.value+step)
        b = fun.value
        Niter += 1
    root = [v.value for v in var]
    res = np.sqrt(sum([v**2 for v in f.value]))
    return root,res,Niter

""" Root-finding example """

x = Var()
y = Var()
f = Array([x**2 + y**2 - 1,
           (3*x - y)**2 + x**2 - 1])

# test four different initial points to locate each of 4 roots
g1 = [0.5,2.]
g2 = [-0.5,-2.]
g3 = [0.5,0.5]
g4 = [-0.5,-0.5]
for g in [g1,g2,g3,g4]:
    print("Initial guess:",g)
    print("----------------------")
    root, res, Niter = newton_multidim(f,[x,y],g)
    print("Converged to root:",[float('%.8f'%r) for r in root])
    print("2-norm of residual:",res)
    print("Number of iterations:",Niter)
    print("\n")

print("Exact solutions")
print("---------------")
b1 = np.array([0,1])
b2 = np.array([0,-1])
b3 = np.array([2.0/np.sqrt(13),3.0/np.sqrt(13)])
b4 = np.array([-2.0/np.sqrt(13),-3.0/np.sqrt(13)])
for b in [b1,b2,b3,b4]:
    print(b)

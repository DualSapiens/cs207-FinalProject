

from ..bfgs import BFGS
from gradpy.autodiff import Var, Array
import numpy as np
import pytest


# Rosenbrock test function
def Rosen(x, y):
    a = y-x*x
    b = 1.-x
    return 100.*a*a+b*b


class TestBFGS:
    def test_Rosen(self):
        x = Var()
        y = Var()
        f = Rosen(x, y)
        # test different initial conditions
        step, Niter, found = BFGS(f, [x, y], [-1., 1.])
        assert np.isclose(x.value, 1)
        assert np.isclose(y.value, 1)
        step, Niter, found = BFGS(f, [x, y], [0., 1.])
        assert np.isclose(x.value, 1)
        assert np.isclose(y.value, 1)
        step, Niter, found = BFGS(f, [x, y], [2., 1.])
        assert np.isclose(x.value, 1)
        assert np.isclose(y.value, 1)

    def test_maxiter(self):
        x = Var()
        y = Var()
        f = Rosen(x, y)
        step, Niter, found = BFGS(f, [x, y], [2., 1.], maxiter=10)
        assert found is False

    def test_vector_valued(self):
        x = Var()
        y = Var()
        f = Array([x**2, y**2])
        with pytest.raises(Exception):
            step, Niter, found = BFGS(f, [x, y], [-1., 1.])

    def test_univariate(self):
        x = Var()
        f = x**4
        step, Niter, found = BFGS(f, x, -3., tol=1e-12)
        assert np.isclose(x.value, 0)
        assert np.isclose(f.value, 0)

import sys
sys.path.append("../../../gradpy")

from ..bfgs import BFGS
from gradpy.autodiff import Var
import numpy as np

# Rosenbrock test function
def Rosen(x,y):
    a = y-x*x
    b = 1.-x
    return 100.*a*a+b*b

class TestBFGS:
    def test_bfgs_Rosen(self):
        x = Var()
        y = Var()
        f = Rosen(x,y)
        # test different initial conditions
        step, Niter = BFGS(f,[x,y],[-1.,1.])
        assert np.isclose(x.value,1)
        assert np.isclose(y.value,1)
        step, Niter = BFGS(f,[x,y],[0.,1.])
        assert np.isclose(x.value,1)
        assert np.isclose(y.value,1)
        step, Niter = BFGS(f,[x,y],[2.,1.])
        assert np.isclose(x.value,1)
        assert np.isclose(y.value,1)

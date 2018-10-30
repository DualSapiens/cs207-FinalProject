import pytest
import numpy as np
from ..autodiff import *


class Test_Module:
    def test_initialization(self):
        x = Var(value=5)
        y = Var(value=3)
        assert x.value == 5
        assert y.value == 3

    def test_addition(self):
        x = Var(value=5)
        f = 3+x
        assert f.der(x) == 1

    def test_multivar_addition(self):
        x = Var(value=5)
        y = Var(value=3)
        f = x + y
        assert f.der(x) == 1
        assert f.der(y) == 1

    def test_subtraction(self):
        x = Var(value=5)
        f = 3-x
        assert f.der(x) == -1
    
    def test_multivar_subtraction(self):
        x = Var(value=5)
        y = Var(value=3)
        f = x - y
        assert f.der(x) == 1
        assert f.der(y) == -1

    def test_mul(self):
        x = Var(value=5)
        f = 3*x
        assert f.der(x) == 3

    def test_multivar_mul(self):
        x = Var(value=5)
        y = Var(value=3)
        f = x * y
        assert f.der(x) == 3
        assert f.der(y) == 5

    def test_div(self):
        x = Var(value=5)
        f = 3/x
        assert f.der(x) == -0.12

    def test_multivar_div(self):
        x = Var(value=5)
        y = Var(value=3)
        f = y / x
        assert f.der(x) == -0.12
        assert f.der(y) == 0.2

    def test_power(self):
        x = Var(value=5)
        f = x**3
        assert f.der(x) == 75

    def test_rpower(self):
        x = Var(value=3)
        f = 5**x
        assert f.der(x) == 201.17973905426254

    def test_multivar_power(self):
        x = Var(value=5)
        y = Var(value=3)
        f = x ** y
        assert f.der(x) == 75
        assert f.der(y) == 201.17973905426254

    def test_pos(self):
        x = Var(value=5)
        f = +x
        assert f.der(x) == 1

    def test_neg(self):
        x = Var(value=5)
        f = -x
        assert f.der(x) == -1

    def test_sin(self):
        x = Var(value=np.pi/2)
        f = Sin(x)
        assert np.isclose(f.der(x), 0)

    def test_cos(self):
        x = Var(value=np.pi/2)
        f = Cos(x)
        assert np.isclose(f.der(x), -1)

    def test_delayed_value_assignment(self):
        x = Var()
        f = Cos(x)
        with pytest.raises(Exception):
            f.der(x)
        x.set_value(np.pi/2)
        assert np.isclose(f.der(x), -1)

    def test_changed_value_assignment(self):
        x = Var(value=0)
        f = Cos(x)
        assert np.isclose(f.der(x), 0)
        x.set_value(np.pi / 2)
        assert np.isclose(f.der(x), -1)

    def test_exp(self):
        x = Var(value=5)
        f = Exp(x)
        assert f.der(x) == 148.4131591025766

    def test_exp_const(self):
        x = Var(value=5)
        f = x * Exp(5)
        assert f.der(x) == 148.4131591025766

    def test_log(self):
        x = Var(value=5)
        f = Log(x)
        assert f.der(x) == 0.2

    def test_log_base(self):
        x = Var(value=8)
        y = Var(value=2)
        f = Log(x,base=y)
        assert f.value == 3
        assert f.der(x) == 0.18033688011112042
        assert f.der(y) == -2.1640425613334453

    def test_sqrt(self):
        x = Var(value=4)
        f = Sqrt(x)
        assert f.value == 2
        assert f.der(x) == 0.25

    def test_addition_of_sin_and_cos(self):
        x1 = Var(3)
        x2 = Var(4)
        f = Sin(x1) + Cos(x2)
        assert f.der(x1) == -0.9899924966004454

    def test_array(self):
        x1 = Var(3)
        x2 = Var(4)
        f = Array([3*x1**2 + 2*x2,
                   Sin(x1) + Cos(x2),
                   Exp(x1**3)])
        assert f.der(x1) == [18,
                             -0.9899924966004454,
                             14365302496248.562
                             ]
        assert f.der(x2) == [2,
                             0.7568024953079282,
                             0]

    def test_doc(self):
        x = Var()
        y = Var()
        f = 5 * x ** 2 + 3 * y
        x.set_value(5)
        y.set_value(3)
        assert f.der(x) == 50

import pytest
import numpy as np
from ..autodiff import *
from ..math import *


class Test_Module_Basic:

    def test_initialization(self):
        x = Var(value=5)
        y = Var(value=3)
        assert x.value == 5
        assert y.value == 3  

    def test_addition(self):
        x = Var(value=5)
        f = 3+x
        assert f.der(x) == 1

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

    def test_sinh(self):
        x = Var(value=0)
        f = Sinh(x)
        assert np.isclose(f.value, 0)
        assert np.isclose(f.der(x), 1)

    def test_cosh(self):
        x = Var(value=0)
        f = Cosh(x)
        assert np.isclose(f.value, 1)
        assert np.isclose(f.der(x), 0)

    def test_tanh(self):
        x = Var(value=0)
        f = Tanh(x)
        assert np.isclose(f.value, 0)
        assert np.isclose(f.der(x), 1)

    def test_cos(self):
        x = Var(value=np.pi/2)
        f = Cos(x)
        assert np.isclose(f.der(x), -1)

    def test_tan(self):
        x = Var(value=0)
        f_1 = Tan(x)
        assert np.isclose(f_1.der(x), 1)
        
    def test_arcsin(self):
        x = Var(value=1)
        f = Arcsin(x)
        assert np.isclose(f.value, np.pi/2)

    def test_arccos(self):
        x = Var(value=0)
        f = Arccos(x)
        assert np.isclose(f.value, np.pi/2)

    def test_arctan(self):
        x = Var(value=0)
        f = Arctan(x)
        assert np.isclose(f.value, 0)

    def test_arcsin_der(self):
        x = Var(value=0.8)
        f = Arcsin(x)
        assert np.isclose(f.der(x), 5/3)

    def test_arccos_der(self):
        x = Var(value=0.8)
        f = Arccos(x)
        assert np.isclose(f.der(x), -5/3)

    def test_arctan_der(self):
        x = Var(value=0.8)
        f = Arctan(x)
        assert np.isclose(f.der(x), 25/41)

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

    def test_logistic(self):
        x = Var(8)
        f = Logistic(x, 2, 3, 0.2)
        assert f.value == 2.305574350497053
        assert f.der(x) == 0.10673666438808344

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
        assert np.all(f.der(x1) == [18,
                                    -0.9899924966004454,
                                    14365302496248.562
                                    ])
        assert np.all(f.der(x2) == [2,
                                    0.7568024953079282,
                                    0])

    def test_eq(self):
        x1 = Var(3)
        x2 = Var(3)
        x3 = Var(4)
        a = (x1 == x2)
        b = (x1 == x3)
        assert a is True
        assert b is False

    def test_ne(self):
        x1 = Var(3)
        x2 = Var(3)
        x3 = Var(4)
        a = (x1 != x2)
        b = (x1 != x3)
        assert a is False
        assert b is True

    def test_lt(self):
        x1 = Var(3)
        x2 = Var(3)
        x3 = Var(4)
        a = (x1 < x2)
        b = (x1 < x3)
        assert a is False
        assert b is True

    def test_gt(self):
        x1 = Var(3)
        x2 = Var(3)
        x3 = Var(4)
        a = (x1 > x2)
        b = (x1 > x3)
        assert a is False
        assert b is False

    def test_le(self):
        x1 = Var(3)
        x2 = Var(3)
        x3 = Var(4)
        a = (x1 <= x2)
        b = (x1 <= x3)
        assert a is True
        assert b is True

    def test_ge(self):
        x1 = Var(3)
        x2 = Var(3)
        x3 = Var(4)
        a = (x1 >= x2)
        b = (x1 >= x3)
        assert a is True
        assert b is False


class Test_Module_Multivariate:

    def test_multivar_grad(self):
        x = Var(value=5)
        y = Var(value=3)
        f = 2*x**3 + 3*y**2
        assert np.all(f.grad([x,y]) == [150, 18])

    def test_array_grad(self):
        x = Var(value=5)
        y = Var(value=2)
        f = Array([2*x**3 + 3*y**2,
                   5./x - 2./y**2])
        assert np.all(f.grad([x,y]) == [[150, 12],
                                        [-0.2, 0.5]])

    def test_array_getitem(self):
        x = Var(value=5)
        y = Var(value=2)
        f = Array([2*x**3 + 3*y**2,
                   5./x - 2./y**2])
        assert len(f) == 2
        assert np.all(f.grad([x,y]) == [[f[0].der(x), f[0].der(y)],
                                        [f[1].der(x), f[1].der(y)]])

    def test_array_setitem(self):
        x = Var()
        y = Var()
        f = Array([0,0])
        with pytest.raises(Exception):
            f[0] = [2*x, 3*y]
        f[0] = 2*x**3 + 3*y**2
        f[1] = 5./x - 2./y**2
        x.set_value(5)
        y.set_value(2)
        assert np.all(f.grad([x,y]) == [[150, 12],
                                        [-0.2, 0.5]])

    def test_array_setitem_const(self):
        x = Var(value=5)
        y = Var(value=2)
        f = Array([0,0])
        f[0] = 2*x**3 + 3*y**2
        f[1] = 5
        assert np.all(f.grad([x,y]) == [[150, 12],
                                        [0, 0]])

class Test_Module_Multiple_Operations:
    def test_multivar_addition(self):
        x = Var(value=5)
        y = Var(value=3)
        f = x + y
        assert f.der(x) == 1
        assert f.der(y) == 1
        
    def test_doc(self):
        x = Var()
        y = Var()
        f = 5 * x ** 2 + 3 * y
        x.set_value(5)
        y.set_value(3)
        assert f.der(x) == 50
    
    #test addition and subtraction:
    def test_addition_subtraction_combined(self):
        x = Var(value=5)
        y = Var(value=3)
        z= Var(value=4)
        f = x - y + z
        assert f.value == 6
        assert f.der(x) == 1
        assert f.der(y) == -1
        assert f.der(z) == 1
        
    def test_addition_subtraction_multiplication_combined(self):
        x = Var(value=5)
        y = Var(value=3)
        z= Var(value=4)
        w = Var(value=2)
        f = (2*x) - y + z * w
        assert f.value == 15
        assert f.der(w) == 4
        assert f.der(x) == 2
        assert f.der(y) == -1
        assert f.der(z) == 2        
        
class Test_Module_Other:        
     def test_string(self):
        x = Var(value="CAT")
        y = Var(value=3)
        assert x.value == "CAT"
        assert y.value == 3

Usage
========
The following three examples illustrate some common uses of the `autodiff` package.

Univariate, scalar-valued function
-----------------------------------
After importing `autodiff`, a `Var` object is created which represents an independent variable. A function f defined in terms of this object will be differentiated, and its derivative with respect to x accessed through the method `der(x)`.

.. code-block:: python

    from autodiff import autodiff

    x = autodiff.Var()

    f = 5*x**2

    x.set_value(5)
    print(f.der(x)) # prints the derivative of f with respect to x

Multivariate, scalar-valued function
--------------------------------------
A multivariable function is defined by initializing more than one `Var` object. Then, the derivative with respect to each independent variable may be accessed by passing in the appropriate variable in the argument of the `der()` method. The `grad()` method allows several derivatives to be returned at once. Here we also illustrate the use of special functions `Sin` and `Cos`, imported from the `autodiff` math module.

.. code-block:: python

    from autodiff import autodiff
    from autodiff import math as admath

    x = autodiff.Var()
    y = autodiff.Var()

    f = admath.Sin(x)**2 + 2*admath.Cos(y)

    x.set_value(5)
    y.set_value(3)
    print(f.der(x)) # prints the derivative of f with respect to x
    print(f.der(y)) # prints the derivative of f with respect to y
    assert np.all(f.grad([x,y]) == [f.der(x), f.der(y)])

Multivariate, vector-valued function
--------------------------------------
Vector-valued functions can be defined using the `Array` class of `autodiff`. Then, the `der.()` method returns a list of derivatives whose length is equal to the number of components of f, while the `grad()` method can return a list of lists - the complete Jacobian of the function.

.. code-block:: python

    from autodiff import autodiff

    x = autodiff.Var()
    y = autodiff.Var()

    f = autodiff.Array([5*x**2+3*y,
                    3*x+2*y**2])

    x.set_value(5)
    y.set_value(3)
    assert np.all(f.der(x) == [f[0].der(x),
                    f[1].der(x)])

    assert np.all(f.grad([x,y]) == [[f[0].der(x), f[0].der(y)],
                         [f[1].der(x), f[1].der(y)]])
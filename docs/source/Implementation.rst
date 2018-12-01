Implementation
=================
Core Data Structures
---------------------
- A list for unique IDs that identify operations

Core Classes
--------------
The `Operation` Class
^^^^^^^^^^^^^^^^^^^^^^
The `Operation` Class forms the core data structure for automatic differentiation. Subclasses of the `Operation` include `Var`, `Constant`, operators, and elementary functions. Its attributes and methods are the following:

.. code-block:: python

    Class Operation:
        Attributes:
            ID
            _value
        Methods:
            __eq__(self, other)
            __ne__(self, other)
            __lt__(self, other)
            __gt__(self, other)
            __le__(self, other)
            __ge__(self, other)
            __add__(self, other)
            __radd__(self, other)
            __sub__ (self, other)
            __rsub__ (self, other)
            __mul__(self, other)
            __rmul__(self, other)
            __truediv__ (self, other)
            __rtruediv__ (self, other)
            __pow__(self, power, modulo=None)
            __rpow__(self, other)
            __pos__(self)
            __neg__ (self)
            evaluate(self)
            der(self, op)
            grad(self, ops)


The `Var` Subclass
^^^^^^^^^^^^^^^^^^^^^^
Instances of the `Var` subclass represent independent variables. User-defined functions that can be differentiated by our package are expressed in terms of `Var` instances. The `Var` Class inherits from its `Operation` superclass, and defines an additional method `set_value` to support delayed value assignment when a value is not supplied on initialization:

.. code-block:: python

    Class Var(Operation):
        Methods:
            set_value(self, value)

The `Constant` Subclass
^^^^^^^^^^^^^^^^^^^^^^^^^^
The `Constant` Subclass allows our package to handle constants in a user-defined function.

Elementary Operations Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Elementary operations include: `Addition`, `Multiplication`, `Subtraction`, `Division`, `Power`, `Pos`, and `Neg`, and overload the existing operations. They are defined as subclasses of `Operation`. User-defined functions involving these operations will be instances of an operation subclass, corresponding to the operation applied at the last step in the computational graph.

Elementary Functions Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Trigonometric functions: `Sin`, `Cos`, `Tan`
- Inverse trigonometric functions: `Arcsin`, `Arccos`, `Arctan`
- Exponentials: `Exp`
- Hyperbolic functions: `Sinh`, `Cosh`, `Tanh`
- Logistic function: `Logistic`
- Logarithms: `Log`
- Square root: `Sqrt`

A user wishing to define a function with these operations must import them from the `gradpy.math` module prior to defining the function. As with elementary operations, user-defined functions using elementary functions will be instances of the elementary function if it is the last step applied in the computational graph. An external dependency of our package is `numpy`, which allows us to carry out the elementary operations within our subclasses.

The `Array` Class
^^^^^^^^^^^^^^^^^^^^
The `Array` Class is used to define vector-valued functions. A user wishing to define a vector valued function can express their function as an instance of `Array`, which takes as its argument a list of the components of the function. It includes methods to collectively return the values, derivatives, and gradient of its components as follows:

.. code-block:: python

    Array:
        Attributes:
            value
        Methods:
            der(self, var_op)
            grad(self, var_ops)

An initialized instance of the `Array` class supports indexing and entry re-assignment (through `__getitem__` and `__setitem__`). Note that instances of the `Array` Class are not themselves `Operation` objects; rather, the components of an `Array` instance are `Operation` objects.

The `IDAllocator` Class
^^^^^^^^^^^^^^^^^^^^^^^^^
The `IDAllocator` Class is a helper class that allocates a unique ID to operations and variables. Its attributes and methods are the following:

.. code-block:: python

    IDAllocator:
        Attributes:
            ids
        Methods:
            allocate_id(cls)

External Dependencies
-----------------------
`NumPy`_ - A Fundamental Package for Scientific Computing with Python.

.. _NumPy: http://www.numpy.org/

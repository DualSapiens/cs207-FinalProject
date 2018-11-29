import numpy as np
from .autodiff import Operation, Constant


class Exp(Operation):
    """Natural exponential function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return self.value * self.op.der(op)

    def evaluate(self):
        self._value = np.exp(self.op.value)


class Log(Operation):
    """Logarithm (default Natural Logarithm)
    """

    def __init__(self, op, base=Constant(np.e), ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)
        self.base = base if isinstance(base, Operation) else Constant(base)

    def der(self, op):
        derivative = 0
        base_value = self.base.value
        op_value = self.op.value
        derivative += (np.log(base_value) / op_value *
                       self.op.der(op)) / (np.log(base_value)) ** 2
        derivative -= (np.log(op_value) / base_value *
                       self.base.der(op)) / (np.log(base_value)) ** 2
        return derivative

    def evaluate(self):
        self._value = np.log(self.op.value) / np.log(self.base.value)


class Logistic(Operation):
    """Logistic function f(X) = L/(1+exp(-k(X-x_0)))
        default: sigmoid function
    """

    def __init__(self, op, x_0 = 0, L = 1, k = 1, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)
        self.x_0 = x_0 if isinstance(x_0, Operation) else Constant(x_0)
        self.L = L if isinstance(L, Operation) else Constant(L)
        self.k = k if isinstance(k, Operation) else Constant(k)

    def der(self, op):
        op_value = self.op.value
        x_0_value = self.x_0.value
        k_value = self.k.value
        L_value = self.L.value
        return k_value * L_value * np.exp(-k_value * (op_value - x_0_value)) \
               * np.power(1 + np.exp(-k_value * (op_value - x_0_value)), -2) \
               * self.op.der(op)

    def evaluate(self):
        self._value = self.L.value / (1 + np.exp(-self.k.value * (self.op.value - self.x_0.value)))


class Sqrt(Operation):
    """Square root
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return 1. / (2 * self.value) * self.op.der(op)

    def evaluate(self):
        self._value = np.sqrt(self.op.value)


class Sin(Operation):
    """Sine function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return np.cos(self.op.value) * self.op.der(op)

    def evaluate(self):
        self._value = np.sin(self.op.value)


class Arcsin(Operation):
    """Inverse sine function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return (1 / np.sqrt(1 - np.square(self.op.value))) * self.op.der(op)

    def evaluate(self):
        self._value = np.arcsin(self.op.value)


class Sinh(Operation):
    """Hyperbolic Sine function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return np.cosh(self.op.value) * self.op.der(op)

    def evaluate(self):
        self._value = np.sinh(self.op.value)


class Cos(Operation):
    """Cosine function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return -np.sin(self.op.value) * self.op.der(op)

    def evaluate(self):
        self._value = np.cos(self.op.value)


class Arccos(Operation):
    """Inverse cosine function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return -(1 / np.sqrt(1 - np.square(self.op.value))) * self.op.der(op)

    def evaluate(self):
        self._value = np.arccos(self.op.value)


class Cosh(Operation):
    """Hyperbolic cosine function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return np.sinh(self.op.value) * self.op.der(op)

    def evaluate(self):
        self._value = np.cosh(self.op.value)


class Tan(Operation):
    """Tangent function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return (1 / np.square(np.cos(self.op.value))) * self.op.der(op)

    def evaluate(self):
        self._value = np.tan(self.op.value)


class Arctan(Operation):
    """Inverse tangent function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return (1 / (1 + np.square(self.op.value))) * self.op.der(op)

    def evaluate(self):
        self._value = np.arctan(self.op.value)


class Tanh(Operation):
    """Hyperbolic tangent function
    """

    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        return (1-np.square(self.value)) * self.op.der(op)

    def evaluate(self):
        self._value = np.tanh(self.op.value)

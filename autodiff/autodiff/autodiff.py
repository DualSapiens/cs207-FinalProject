# Implementation of Automatic Differentiation
import numpy as np


class IDAllocator:
    ids = []

    @classmethod
    def allocate_id(cls):
        if len(cls.ids) == 0:
            ID = 0
            cls.ids.append(ID)
        else:
            ID = cls.ids[-1]+1
            cls.ids.append(ID)
        return ID


class Operation:
    def __init__(self, value=None, ID=None):
        if ID is None:
            ID = IDAllocator.allocate_id()
        self.ID = ID
        self._value = value

    @property
    def value(self):
        self.evaluate()
        return self._value

    def __eq__(self, other):
        return self.ID == other.ID

    def __add__(self, other):
        if isinstance(other, Operation):
            return Addition(self, other)
        else:
            return Addition(self, Constant(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Operation):
            return Subtraction(self, other)
        else:
            return Subtraction(self, Constant(other))

    def __rsub__(self, other):
        return Constant(other) - self

    def __mul__(self, other):
        if isinstance(other, Operation):
            return Multiplication(self, other)
        else:
            return Multiplication(self, Constant(other))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Operation):
            return Division(self, other)
        else:
            return Division(self, Constant(other))

    def __rtruediv__(self, other):
        return Constant(other) / self

    def __pow__(self, power, modulo=None):
        if isinstance(power, Operation):
            return Power(self, power)
        else:
            return Power(self, Constant(power))

    def __rpow__(self, other):
        if isinstance(other, Operation):
            return Power(other, self)
        else:
            return Power(Constant(other), self)
    
    def __pos__(self):
        return Pos(self)

    def __neg__(self):
        return Neg(self)

    def evaluate(self):
        raise NotImplementedError

    def der(self, op):
        raise NotImplementedError

    def grad(self, ops):
        return np.array([self.der(op) for op in ops])


class Var(Operation):
    def __init__(self, value=None, ID=None):
        super().__init__(value, ID)

    def set_value(self, value):
        self._value = value

    def evaluate(self):
        if self._value is None:
            raise Exception("Evaluate ")
        return self._value

    def der(self, op):
        if self == op:
            return 1
        else:
            return 0


class Constant(Operation):
    def __init__(self, value=None, ID=None):
        if value is None:
            raise Exception("Cannot have not-valued constant")
        super().__init__(value=value, ID=ID)

    def evaluate(self):
        return self._value

    def der(self, op):
        return 0

class Addition(Operation):
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        self.evaluate()
        derivative = 0
        derivative += self.op_1.der(op)
        derivative += self.op_2.der(op)
        return derivative

    def evaluate(self):
        self._value = self.op_1.value + self.op_2.value

class Subtraction(Operation):
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        self.evaluate()
        derivative = 0
        derivative += self.op_1.der(op)
        derivative -= self.op_2.der(op)
        return derivative

    def evaluate(self):
        self._value = self.op_1.value - self.op_2.value

class Multiplication(Operation):
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        self.evaluate()
        derivative = 0
        derivative += self.op_2.value * self.op_1.der(op)
        derivative += self.op_1.value * self.op_2.der(op)
        return derivative

    def evaluate(self):
        self._value = self.op_1.value * self.op_2.value

class Division(Operation):
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        self.evaluate()
        derivative = 0
        derivative += (self.op_2.value * self.op_1.der(op))/(self.op_2.value)**2
        derivative -= (self.op_1.value * self.op_2.der(op))/(self.op_2.value)**2
        return derivative

    def evaluate(self):
        self._value = self.op_1.value / self.op_2.value

class Power(Operation):
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        self.evaluate()
        derivative = 0
        derivative += self.op_2.value * (self.op_1.value ** (
                self.op_2.value-1)) * self.op_1.der(op)
        if self.op_2.der(op) != 0 and self.value != 0:
            # self.op_1.value must be > 0 here!
            derivative +=  self.value * np.log(self.op_1.value) * self.op_2.der(op)
        return derivative

    def evaluate(self):
        self._value = self.op_1.value**self.op_2.value

class Pos(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op

    def der(self, op):
        self.evaluate()
        return self.op.der(op)

    def evaluate(self):
        self._value = self.op.value

class Neg(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op

    def der(self, op):
        self.evaluate()
        return -self.op.der(op)

    def evaluate(self):
        self._value = -self.op.value

class Array:
    def __init__(self, ops):
        self.ops = ops

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, index):
        return self.ops[index]

    def __setitem__(self, index, op):
        try:
            l = len(op)
            raise Exception("Cannot set Array value with an array or list.")
        except TypeError:
            if isinstance(op, Operation):
                self.ops[index] = op
            else:
                self.ops[index] = Constant(op)

    @property
    def value(self):
        values = np.array([op.value for op in self.ops])
        return values

    def der(self, var_op):
        derivatives = np.array([op.der(var_op) for op in self.ops])
        return derivatives

    def grad(self, var_ops):
        gradients = np.array([op.grad(var_ops) for op in self.ops])
        return gradients

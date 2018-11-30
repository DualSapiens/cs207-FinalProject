# Implementation of Automatic Differentiation
import numpy as np


class IDAllocator:
    """Allocate unique IDs for different operations
    """

    # list of id for created operations
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
    """Super-class of all the elementary operations/functions as well as
       variables and constants
    """
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
        if isinstance(other, Operation):
            if other._value is not None:
                return self._value == other._value
            else:
                raise Exception('The operation does not have a value.')
        else:
            raise TypeError('Value comparison must be between two Operation objects.')


    def __ne__(self, other):
        if isinstance(other, Operation):
            if other._value is not None:
                return self._value != other._value
            else:
                raise Exception('The operation does not have a value.')
        else:
            raise TypeError('Value comparison must be between two Operation objects.')

    def __lt__(self, other):
        if isinstance(other, Operation):
            if other._value is not None:
                return self._value < other._value
            else:
                raise Exception('The operation does not have a value.')
        else:
            raise TypeError('Value comparison must be between two Operation objects.')

    def __gt__(self, other):
        if isinstance(other, Operation):
            if other._value is not None:
                return self._value > other._value
            else:
                raise Exception('The operation does not have a value.')
        else:
            raise TypeError('Value comparison must be between two Operation objects.')

    def __le__(self, other):
        if isinstance(other, Operation):
            if other._value is not None:
                return self._value <= other._value
            else:
                raise Exception('The operation does not have a value.')
        else:
            raise TypeError('Value comparison must be between two Operation objects.')


    def __ge__(self, other):
        if isinstance(other, Operation):
            if other._value is not None:
                return self._value >= other._value
            else:
                raise Exception('The operation does not have a value.')
        else:
            raise TypeError('Value comparison must be between two Operation objects.')


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
        return self

    def __neg__(self):
        return Neg(self)


    def evaluate(self):
        """ Evaluate the value of the entire operation

        stores the value internally
        """
        raise NotImplementedError

    def der(self, op):
        """Compute the derivative of the operation with respect to a variable

        :param op: the variable to take derivative against
        :return: the derivative value
        """
        raise NotImplementedError

    def grad(self, ops):
        """Compute the gradient of the operation with respect to a set of variables

        :param ops: the variables to take gradient against
        :return: the gradient as a np array
        """
        return np.array([self.der(op) for op in ops])




class Var(Operation):
    """Represent a scalar variable
    """
    def __init__(self, value=None, ID=None):
        super().__init__(value, ID)

    def set_value(self, value):
        """Set/change the value of the variable

        :param value: the value to set to
        """
        self._value = value

    def evaluate(self):
        if self._value is None:
            raise Exception("Variable value not set yet")
        return self._value

    def der(self, op):
        if self.ID == op.ID:
            return 1
        else:
            return 0



class Constant(Operation):
    """Represent a scalar constant
    """
    def __init__(self, value=None, ID=None):
        if value is None:
            raise Exception("Cannot have not-valued constant")
        super().__init__(value=value, ID=ID)

    def evaluate(self):
        pass

    def der(self, op):
        return 0


class Addition(Operation):
    """Addition between two ops
    """
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        derivative = self.op_1.der(op)
        derivative += self.op_2.der(op)
        return derivative

    def evaluate(self):
        self._value = self.op_1.value + self.op_2.value


class Subtraction(Operation):
    """First op subtracted by second op
    """
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        derivative = self.op_1.der(op)
        derivative -= self.op_2.der(op)
        return derivative

    def evaluate(self):
        self._value = self.op_1.value - self.op_2.value


class Multiplication(Operation):
    """Multiply two ops
    """
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        derivative = self.op_2.value * self.op_1.der(op)
        derivative += self.op_1.value * self.op_2.der(op)
        return derivative

    def evaluate(self):
        self._value = self.op_1.value * self.op_2.value


class Division(Operation):
    """first op divided by second op
    """
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        op_2_value = self.op_2.value
        derivative = (op_2_value * self.op_1.der(op))/(op_2_value)**2
        derivative -= (self.op_1.value * self.op_2.der(op))/(op_2_value)**2
        return derivative

    def evaluate(self):
        self._value = self.op_1.value / self.op_2.value


class Power(Operation):
    """Raise first op to the power of second op
    """
    def __init__(self, op_1, op_2, ID=None):
        super().__init__(ID=ID)
        self.op_1 = op_1
        self.op_2 = op_2

    def der(self, op):
        derivative = 0
        op_2_value = self.op_2.value
        op_1_value = self.op_1.value
        op_2_der = self.op_2.der(op)
        self._value = op_1_value**op_2_value
        derivative += op_2_value * (op_1_value ** (
                op_2_value-1)) * self.op_1.der(op)
        if op_2_der != 0 and self.value != 0:
            # self.op_1.value must be > 0 here!
            derivative +=  self._value * np.log(op_1_value) * op_2_der
        return derivative

    def evaluate(self):
        self._value = self.op_1.value**self.op_2.value


class Neg(Operation):
    """Negation
    """
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op

    def der(self, op):
        return -self.op.der(op)

    def evaluate(self):
        self._value = -self.op.value


class Array:
    """Represent vector functions
    """
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

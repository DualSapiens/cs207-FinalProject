import numpy as np
from .autodiff import Operation, Constant

class Exp(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        self.evaluate()
        return self.value*self.op.der(op)

    def evaluate(self):
        self._value = np.exp(self.op.value)

class Log(Operation):
    def __init__(self, op, base=Constant(np.e), ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)
        self.base = base if isinstance(base, Operation) else Constant(base)

    def der(self, op):
        self.evaluate()
        derivative = 0
        derivative += (np.log(self.base.value) / self.op.value *
                self.op.der(op)) / (np.log(self.base.value))**2
        derivative -= (np.log(self.op.value) / self.base.value *
                self.base.der(op)) / (np.log(self.base.value))**2
        return derivative

    def evaluate(self):
        self._value = np.log(self.op.value) / np.log(self.base.value)

class Sqrt(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        self.evaluate()
        return 1./(2*self.value)*self.op.der(op)

    def evaluate(self):
        self._value = np.sqrt(self.op.value)

class Sin(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        self.evaluate()
        return np.cos(self.op.value)*self.op.der(op)

    def evaluate(self):
        self._value = np.sin(self.op.value)
        
class Arcsin(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        self.evaluate()
        return (1/np.sqrt(1-np.square(self.op.value)))*self.op.der(op)

    def evaluate(self):
        self._value = np.arcsin(self.op.value)


class Arccos(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        self.evaluate()
        return -(1/np.sqrt(1-np.square(self.op.value)))*self.op.der(op)

    def evaluate(self):
        self._value = np.arccos(self.op.value)


class Cos(Operation):
    def __init__(self, op, ID=None):
        super().__init__(ID=ID)
        self.op = op if isinstance(op, Operation) else Constant(op)

    def der(self, op):
        self.evaluate()
        return -np.sin(self.op.value)*self.op.der(op)

    def evaluate(self):
        self._value = np.cos(self.op.value)

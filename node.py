import math
import random
from abc import ABC, abstractmethod

class Operation(ABC):
    def __init__(self, label) -> None:
        self.label = label
    
    @abstractmethod
    def __call__(self, data1, data2):
        raise NotImplementedError
    
    @abstractmethod
    def _backward(self, grad_out):
        raise NotImplementedError

class OperationFactory:
    def __call__(self, op, node1, node2 = None):
        self.op = op
        
        if node2:
            if not isinstance(node2, Value):
                node2 = Value(node2)
            
            def _backward():
                grad1_update, grad2_update = self.op._backward(out.grad)
                node1.grad += grad1_update
                node2.grad += grad2_update
            
            out = Value(self.op(node1.data, node2.data), _childern = (node1, node2), _op = op.label)
            out._backward = _backward
        else:
            def _backward():
                grad1_update, _ = self.op._backward(out.grad)
                node1.grad += grad1_update
            
            out = Value(self.op(node1.data, None), _childern = (node1, ), _op = op.label)
            out._backward = _backward

        return out

class Addition(Operation):
    def __init__(self, label = "add") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        return data1 + data2
    
    def _backward(self, grad_out):
        return (grad_out, grad_out)

class Multiplication(Operation):
    def __init__(self, label = "mul") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        self.data1, self.data2 = data1, data2
        return data1 * data2

    def _backward(self, grad_out):
        return (grad_out * self.data2, grad_out * self.data1)

class Subtraction(Operation):
    def __init__(self, label = "sub") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        return data1 - data2

    def _backward(self, grad_out):
        return (grad_out, grad_out)

class Division(Operation):
    def __init__(self, label = "div") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        if data2 == 0:
            raise ZeroDivisionError

        self.data1, self.data2 = data1, data2
        return data1 / data2
    
    def _backward(self, grad_out):
        return (grad_out / self.data2, - self.data1 * grad_out / (self.data2 ** 2))

class Power(Operation):
    def __init__(self, label = "pow") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        if not isinstance(data2, (int, float)):
            raise TypeError
        
        self.data1, self.data2 = data1, data2
        return data1 ** data2
    
    def _backward(self, grad_out):
        return (grad_out * self.data2 * self.data1 ** (self.data2 - 1), 0.0)

class Exp(Operation):
    def __init__(self, label = "exp") -> None:
        super().__init__(label)
    
    def __call__(self, data1, data2):
        self.out = math.exp(data1)
        return math.exp(data1)
    
    def _backward(self, grad_out):
        return (self.out * grad_out, None)

class Value:
    def __init__(self, data, label="", _childern = (), _op = '') -> None:
        """
            Value object to store numerical values
            :param data      - numerical value
            :param label     - label for human readability
            :param _childern - all of the childern of the current value node
            :param _op       - operation leading to the current value
        """

        self.data = data
        self.label = label
        self._prev = set(_childern)  # used for backprop (childern is previous)
        self._op = _op
        self.op_fact = OperationFactory()
        self.grad = 0.0  # records the partial derivative of output wrt this node
        self._backward = lambda : None  # used for backpropagation
    
    def __repr__(self) -> str:
        return f"{self.data}"

    def operate(self, other, op):
        out = self.op_fact(op, self, other)
        return out

    def __add__(self, other):
        add = Addition()
        return self.operate(other, add)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        mul = Multiplication()
        return self.operate(other, mul)

    def __rmul__(self, other):  # other * self
        return self * other
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _childern=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out
    
    def __sub__(self, other):
        sub = Subtraction()
        return self.operate(other, sub)

    def __rsub__(self, other):
        return self - other
    
    def __truediv__(self, other):
        division = Division()
        return self.operate(other, division)

    def __pow__(self, other): # self ** other
        pow = Power()
        return self.operate(other, pow)

    def __rpow__(self, other):  # a ^ x
        other = Value(other)
        return other ** self
    
    def exp(self):
        exp = Exp()
        return self.operate(None, exp)

    def backward(self):
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                topo.append(node)  # WHERE IT MIGHT GO WRONG
                for child in node._prev:
                    build_topo(child)
        
        topo = []
        visited = set()
        self.grad = 1.0
        
        build_topo(self)
        for node in topo:
            node._backward()
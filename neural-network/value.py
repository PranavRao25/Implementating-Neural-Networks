import math

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
        self.grad = 0.0  # records the partial derivative of output wrt this node
        self._backward = lambda : None  # used for backpropagation
    
    def __repr__(self) -> str:
        return f"{self.data}"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data + other.data, _childern=(self, other), _op='+')

        def _backward():
            # += to accumulate grads, to avoid overwrites when this node has multiple childern
            self.grad += out.grad  
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data * other.data, _childern=(self, other), _op='*')
    
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

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
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data - other.data, _childern = (self, other), _op = '-')

        def _backward():
            self.grad += out.grad
        
        out._backward = _backward

        return out

    def __rsub__(self, other):
        return self - other
    
    def __div__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        if other.data == 0:
            raise ZeroDivisionError
        
        out = Value(self.data / other.data, _childern = (self, other), _op = '/')

        # out = self / other
        # d(out) = (d(self) * other - self * d(other))/other**2
        # d(out) = d(self) / other - self * d(other) / other ** 2
        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad += - self.data / (other.data ** 2) * out.grad
        
        out._backward = _backward

        return out

    def __pow__(self, other): # self ** other
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data ** other.data, _childern = (self, other), _op = "**")

        # out = self ** other
        # d(out) = other * self ** (other - 1) * d(self) + d(other) * self ** other * ln(self)
        def _backward():
            self.grad += out.grad * other.data * self.data ** (other.data - 1)
            other.grad += out.grad * out.data * math.log(self.data)
        
        out._backward = _backward

        return out

    def __rpow__(self, other):  # a ^ x
        other = Value(other)
        return other ** self
    
    def exp(self):
        out = Value(math.exp(self.data), _childern = (self, ), _op = 'e')

        # out = e^self
        # d(out) = out * d(self)
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

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
import math
import random
from node import *

class Neuron:
    def __init__(self, n_inputs) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum((w.data*i for w,i in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out) -> None:
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out
    
    def parameters(self):
        return [param for n in self.neurons for param in n.parameters()]

class MLP:
    def __init__(self, n_input, n_outs) -> None:
        ins = [n_input] + n_outs
        self.layers = [Layer(ins[i], ins[i+1]) for i in range(len(n_outs))]
    
    def __call__(self, x):
        temp = x
        for layer in self.layers:
            temp = layer(temp)
        return temp
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

# TODO: Documentation

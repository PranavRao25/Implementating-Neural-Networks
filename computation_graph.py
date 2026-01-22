# ============================================================
# Computational Graph & Automatic Backpropagation Implementation
# ============================================================

import numpy as np


# ============================================================
# 1. Define Node class
# Each node represents an operation or variable in the graph.
# ============================================================

class ComputeNode:
    def __init__(self, ntype, idx, *children):
        self.type = ntype              # Type of operation ('var', 'add', 'mul', etc.)
        self.id = idx                  # Unique id (order of creation)
        self.children = list(children) # Parent nodes (inputs to this node)
        self.value = None              # Value after forward pass
        self.grad = None               # Gradient accumulated during backward pass

    def __repr__(self):
        return f"Node({self.id},{self.type})"


# ============================================================
# 2. Define Graph class
# This holds all nodes and provides operations to build the graph.
# ============================================================

class Graph:
    def __init__(self):
        self.nodes = []  # list of nodes in creation (topological) order

    # Create variable node (parameter or input)
    def var(self, value):
        n = ComputeNode('var', len(self.nodes))
        n.value = np.array(value)  # store numeric value
        self.nodes.append(n)
        return n

    # Element-wise or broadcasted addition
    def add(self, a, b):
        v = ComputeNode('add', len(self.nodes), a, b)
        self.nodes.append(v)
        return v

    # Matrix multiplication (for layers)
    def mul(self, a, b):
        v = ComputeNode('mul', len(self.nodes), a, b)
        self.nodes.append(v)
        return v

    # Sigmoid activation function node
    def logistic(self, a):
        v = ComputeNode('logistic', len(self.nodes), a)
        self.nodes.append(v)
        return v

    # Log-Softmax (for stable classification output)
    def logsoftmax(self, pred):
        v = ComputeNode('logsoftmax', len(self.nodes), pred)
        self.nodes.append(v)
        return v

    # Cross-entropy loss node
    def cross_entropy(self, logprob, true):
        v = ComputeNode('cross-entropy', len(self.nodes), logprob, true)
        self.nodes.append(v)
        return v

    # Reset gradients (before a new backward pass)
    def zero_grads(self):
        for n in self.nodes:
            n.grad = None


# ============================================================
# 3. Forward propagation
# Traverses the graph in creation order to compute node values.
# ============================================================

def forward(graph):
    for n in graph.nodes:
        if n.type == 'var':
            # Variables already have their value
            continue

        elif n.type == 'add':
            # Addition (with possible broadcasting)
            a, b = n.children
            n.value = a.value + b.value

        elif n.type == 'mul':
            # Matrix multiplication
            a, b = n.children
            n.value = a.value @ b.value

        elif n.type == 'logistic':
            # Sigmoid activation: s(x) = 1 / (1 + e^-x)
            a, = n.children
            n.value = 1.0 / (1.0 + np.exp(-a.value))

        elif n.type == 'logsoftmax':
            # Log-softmax for numerical stability
            x, = n.children
            shifted = x.value - np.max(x.value, axis=-1, keepdims=True)
            logsum = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            n.value = shifted - logsum  # log(softmax(x))

        elif n.type == 'cross-entropy':
            # Cross-entropy loss: -sum(y * log(p))
            logprob, true = n.children
            n.value = -np.sum(logprob.value * true.value)

        else:
            raise RuntimeError(f"Unknown node type: {n.type}")


# ============================================================
# 4. Backward propagation
# Compute gradients for each node (reverse topological order).
# ============================================================

# Helper: initialize gradient array when needed
def init_grad(n):
    if n.grad is None:
        n.grad = np.zeros_like(n.value)


def backward(graph):
    for n in reversed(graph.nodes):  # reverse traversal (backprop)
        if n.type == 'var':
            # Variables only accumulate gradients; no need to backprop further
            continue

        elif n.type == 'add':
            # ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
            a, b = n.children
            init_grad(a)
            init_grad(b)

            # Handle broadcasting correctly (e.g., bias addition)
            a_grad = n.grad
            b_grad = n.grad

            if a.value.shape != n.value.shape:
                a_grad = np.sum(n.grad, axis=0)
            if b.value.shape != n.value.shape:
                b_grad = np.sum(n.grad, axis=0)

            # Accumulate gradients
            a.grad += a_grad
            b.grad += b_grad

        elif n.type == 'mul':
            # Matrix multiplication: n = a @ b
            # ∂L/∂a = ∂L/∂n @ b.T
            # ∂L/∂b = a.T @ ∂L/∂n
            a, b = n.children
            init_grad(a)
            init_grad(b)
            a.grad += n.grad @ b.value.T
            b.grad += a.value.T @ n.grad

        elif n.type == 'logistic':
            # Sigmoid derivative: s'(x) = s(x) * (1 - s(x))
            a, = n.children
            init_grad(a)
            s = n.value
            a.grad += n.grad * (s * (1.0 - s))

        elif n.type == 'logsoftmax':
            # logsoftmax derivative: ∂L/∂x = g - softmax(x) * sum(g)
            x, = n.children
            init_grad(x)
            g = n.grad
            logp = n.value
            p = np.exp(logp)
            s = np.sum(g * p, axis=-1, keepdims=True)
            x.grad += g - p * s

        elif n.type == 'cross-entropy':
            # dL/dy_hat = -y
            y_hat, true = n.children
            init_grad(y_hat)
            y_hat.grad += -true.value * n.grad

        else:
            raise RuntimeError(f"Unknown node type: {n.type}")


# ============================================================
# 5. Example: 1-hidden-layer Neural Network (1 sample)
# ============================================================

# Random seed for reproducibility
rng = np.random.RandomState(0)

# Input: 1 sample with 3 features
x = rng.randn(1, 3)

# Target class: one-hot encoded for 2 classes
y_idx = 1
y_onehot = np.zeros((1, 2))
y_onehot[0, y_idx] = 1

# Initialize network parameters
W1 = rng.randn(3, 4) * 0.1  # input -> hidden weights
b1 = np.zeros(4)            # hidden bias
W2 = rng.randn(4, 2) * 0.1  # hidden -> output weights
b2 = np.zeros(2)            # output bias

# Build computation graph
g = Graph()
node_W1 = g.var(W1)
node_b1 = g.var(b1)
node_W2 = g.var(W2)
node_b2 = g.var(b2)
node_x = g.var(x)
node_y = g.var(y_onehot[0])

# Forward computation in graph form:
# h = sigmoid(x @ W1 + b1)
# logits = h @ W2 + b2
# logp = logsoftmax(logits)
# loss = cross_entropy(logp, y)
a = g.add(g.mul(node_x, node_W1), node_b1)  # linear + bias
h = g.logistic(a)                           # hidden activation
logits = g.add(g.mul(h, node_W2), node_b2)  # second layer
logp = g.logsoftmax(logits)                 # log probabilities
loss = g.cross_entropy(logp, node_y)        # loss node

# Run forward pass
forward(g)
print("Loss before update:", loss.value)

# Run backward pass (start from loss)
g.zero_grads()
loss.grad = 1.0   # dL/dL = 1, seed gradient
backward(g)

# Inspect gradient shapes
print("W1 grad shape:", node_W1.grad.shape)
print("W2 grad shape:", node_W2.grad.shape)

# Simple gradient descent parameter update
lr = 0.5
node_W1.value -= lr * node_W1.grad
node_b1.value -= lr * node_b1.grad
node_W2.value -= lr * node_W2.grad
node_b2.value -= lr * node_b2.grad

# Forward again after update — loss should decrease
forward(g)
print("Loss after 1 SGD step:", loss.value)
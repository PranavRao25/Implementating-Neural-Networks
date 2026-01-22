from nn import *

def loss_fn(t, p):
    return (t - p) ** 2  # mse

X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
y = [1.0, -1.0, -1.0, 1.0]

n_input = 3
n_outs = [4, 4, 1]
mlp = MLP(n_input, n_outs)

lr = 0.01
n_epochs = 20

for i in range(n_epochs):
    pred = [mlp(x)[0] for x in X]
    loss = sum(loss_fn(t, p) for t, p in zip(y, pred))

    for param in mlp.parameters():
        param.grad = 0  # so that the grad of loss calculated is only for this new iteration
    loss.backward()

    print(f"{i} : {loss.data}")

    for param in mlp.parameters():
        param.data += -lr * param.grad

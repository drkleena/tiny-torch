1. Core idea: separate “engine” from “user API”

Think in 2 layers:

Engine layer = your Value graph, topo sort, backward rules

Library layer = “torch-like” stuff: Tensor, nn.Linear, optim.SGD, etc.

Right now you only have the engine. That’s fine. Let’s propose a layout that grows cleanly.

2. Suggested project layout

Something like:

mytorch/
  autograd/
    __init__.py
    engine.py        # Value, topo sort, primitive ops' backward
    ops.py           # optional: higher-level functions (softmax, cross_entropy, etc.)

  tensor.py          # user-facing Tensor that wraps np.array + hooks into autograd

  nn/
    __init__.py
    modules.py       # Linear, BatchNorm, etc.
    functional.py    # relu(x), tanh(x), softmax(x) style

  optim/
    __init__.py
    sgd.py           # SGD, maybe Adam later

  examples/
    mnist_mlp.py     # your first full training example

  tests/
    test_engine.py
    test_broadcast.py
    test_linear.py
    test_mnist_toy.py


Let me break down the important pieces.

3. autograd/engine.py (what you already have)

Keep this “pure”:

class Value

all primitive ops and their _backward:

+ - * / **

matmul

sum, mean

exp, log, tanh, relu

backward() + topo sort

_unbroadcast helper, etc.

This is your low-level autodiff engine. No MNIST, no layers, no optimizers here.

4. tensor.py — optional but very nice

Right now you’re writing everything directly in terms of Value. For a “torch clone” feel, it’s nice to have a Tensor API as a thin wrapper over Value.

Something like:

from .autograd.engine import Value

class Tensor:
    def __init__(self, data):
        self._v = Value(data)

    @property
    def data(self):
        return self._v.data

    @property
    def grad(self):
        return self._v.grad

    def backward(self):
        self._v.backward()

    # and you can implement __add__, __matmul__, etc. by delegating to Value


You don’t have to do this now, but structuring the project with tensor.py in mind is future-proof.

5. nn/modules.py — your layers

This is where your Linear + activation lives. For now:

from mytorch.autograd.engine import Value
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, activation=None):
        self.W = Value(np.random.randn(in_features, out_features) * 0.01)
        self.b = Value(np.zeros((1, out_features)))
        self.activation = activation

    def __call__(self, x):
        out = x @ self.W + self.b
        if self.activation == "relu":
            out = out.relu()
        elif self.activation == "tanh":
            out = out.tanh()
        return out

    def parameters(self):
        return [self.W, self.b]


And your Network:

class Network:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params


Later you can add BatchNorm, Dropout, etc. here.

6. nn/functional.py — “stateless” ops

Things like:

relu(x)

tanh(x)

softmax(x, axis=-1)

cross_entropy(logits, targets)

Live here rather than in engine.py, so your engine stays minimal and mathy, and the “deep learning” bits live in nn.functional.

You already have:

def softmax(logits: Value, axis=-1):
    ...
def cross_entropy(logits: Value, targets_one_hot: Value):
    ...


Those are perfect candidates for nn/functional.py.

7. optim/sgd.py — one optimizer

For now, even something tiny:

class SGD:
    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.grad)


Then your training loop becomes very PyTorch-y:

optimizer = SGD(model.parameters(), lr=0.1)

for batch in ...:
    logits = model(X)
    loss = cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

8. examples/ — where MNIST lives

Instead of cramming training code into the library, keep it in examples/:

examples/
  mnist_mlp.py      # uses mytorch.nn + mytorch.optim + mytorch.nn.functional


This is where you:

load MNIST with NumPy

build the model

run the loop

print loss/accuracy

9. What to do right now (small next step)

Given what you already have, the next clean move is:

Keep Value + ops in autograd/engine.py as-is.

Create nn/modules.py with:

Linear

Network

Create nn/functional.py with:

softmax

cross_entropy

Move your training code into an examples/mnist_mlp.py.

Then you can gradually add:

optim/sgd.py

BatchNorm1d into nn/modules.py

etc.
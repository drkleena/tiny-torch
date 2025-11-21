#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class Value():

    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = float(data)
        self.grad = float(0)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


# In[3]:


def test_basic_init():
    x = Value(3.5)
    assert isinstance(x.data, float), "data should be a float"
    assert x.data == 3.5, "data should store the input value"
    assert x.grad == 0.0, "grad should start at 0.0"

def test_children_and_metadata():
    a = Value(1.0, label="a")
    b = Value(2.0, label="b")
    c = Value(3.0, _parents=(a, b), op="+", label="c")

    # _prev should contain the parents
    assert a in c._prev and b in c._prev, "_prev should contain parent nodes"

    # op and label stored correctly
    assert c.op == "+", "op should be stored"
    assert c.label == "c", "label should be stored"

    # _backward should be callable
    assert callable(c._backward), "_backward should be a function"

def test_repr():
    x = Value(2.0)
    s = repr(x)
    assert "Value(" in s, "__repr__ should look like Value(...)"
    assert "data=" in s and "grad=" in s, "__repr__ should include data and grad"

if __name__ == "__main__":
    test_basic_init()
    test_children_and_metadata()
    test_repr()
    print("✅ Stage 1 tests passed!")


# ## Scalar Ops

# Addition (+, __add__, __radd__)
# 
# Multiplication (*, __mul__, __rmul__)
# 
# Negation + subtraction (__neg__, __sub__, __rsub__)
# 
# Power (__pow__ with float/int exponent)
# 
# Backward function closures

# In[4]:


class Value():

    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = float(data)
        self.grad = float(0)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__



# In[5]:


a = Value(data=3)
b = Value(data=4)
c = a + b


# In[6]:


a+b


# In[7]:


import math

def test_addition_forward_backward():
    x = Value(2.0)
    y = Value(3.0)
    z = x + y
    z.grad = 1.0
    z._backward()

    print ('x', x, 'y',y, 'z', z)




# In[8]:


test_addition_forward_backward()


# In[9]:


class Value():

    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = float(data)
        self.grad = float(0)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__


# In[10]:


def test_multiplication_forward_backward():
    x = Value(2.0)
    y = Value(3.0)
    z = x * y
    z.grad = 1.0
    z._backward()

    assert z.data == 6.0
    assert x.grad == 3.0
    assert y.grad == 2.0


# In[11]:


test_multiplication_forward_backward()


# In[12]:


class Value():

    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = float(data)
        self.grad = float(0)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    # These do not require backwards, as they are composed of the above primitives

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)    


# In[13]:


def test_neg_and_sub():
    x = Value(5.0)
    y = Value(2.0)

    # negation
    n = -x
    assert n.data == -5.0

    # subtraction
    s = x - y
    assert s.data == 3.0


# In[14]:


test_neg_and_sub()


# ## Power and Division
# 
# 

# In[15]:


class Value():

    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = float(data)
        self.grad = float(0)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    # These do not require backwards, as they are composed of the above primitives

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)    

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, _parents=(self,), op="**")

        def _backward():
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out


    def __truediv__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, _parents=(self, other), op='/')

        def _backward():
            """
            Remember that...
                f = x / y
                df/dx = 1 / y
                df/dy = -x / y^2
            """

            self.grad += 1.0 / (other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad

        other._backward = _backward
        return out

    def __rtruediv__(self, other):
        # other / self
        other = other if isinstance(other, Value) else Value(other)
        return other / self


# ## Backward traversal

# In[16]:


class Value():

    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = float(data)
        self.grad = float(0)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    # These do not require backwards, as they are composed of the above primitives

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)    

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, _parents=(self,), op="**")

        def _backward():
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out


    def __truediv__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, _parents=(self, other), op='/')

        def _backward():
            """
            Remember that...
                f = x / y
                df/dx = 1 / y
                df/dy = -x / y^2
            """

            self.grad += 1.0 / (other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad

        other._backward = _backward
        return out

    def __rtruediv__(self, other):
        # other / self
        other = other if isinstance(other, Value) else Value(other)
        return other / self




    def backward(self):
        """
        Backpropagate gradients from this Value all the way through
        the graph of Values that produced it.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # df/df = 1
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()


# In[17]:


def test_div_grad():
    x = Value(4.0)
    y = Value(2.0)
    z = x / y
    z.backward()

    assert z.data == 2.0  # 2.0
    assert x.grad == 0.5
    assert y.grad == -1.0

    print ("✅ Passed!")


# In[18]:


test_div_grad()


# In[31]:


class Value():

    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = float(data)
        self.grad = float(0)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    # These do not require backwards, as they are composed of the above primitives

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)    

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, _parents=(self,), op="**")

        def _backward():
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out


    def __truediv__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, _parents=(self, other), op='/')

        def _backward():
            """
            Remember that...
                f = x / y
                df/dx = 1 / y
                df/dy = -x / y^2
            """

            self.grad += 1.0 / (other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        # other / self
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def backward(self):
        """
        Backpropagate gradients from this Value all the way through
        the graph of Values that produced it.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # df/df = 1
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()


    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, _parents = (self,), op="tanh")

        def _backward():

            """
            Remember that...
                local gradient = dtanh(x)/dx = 1 - tanh(x)^2
            """
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out



    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, _parents=(self,), op="relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out




# In[32]:


x = Value(1.0)
y = x.tanh()
y.backward()

print(y.data)   # ~0.76159
print(x.grad)   # 1 - tanh(1)^2  ≈ 0.419974

x = Value(2.0)
y = x.relu()
y.backward()

print(y.data)   # 0.0
print(x.grad)   # 0.0


# In[33]:


# sanity check: f(x, y) = x * y + x**2
x = Value(2.0)
y = Value(3.0)

f = x * y + (x ** 2)
f.backward()

print("f:", f.data)        # expect 10.0
print("x.grad:", x.grad)   # df/dx = y + 2x = 3 + 4 = 7
print("y.grad:", y.grad)   # df/dy = x = 2


# In[34]:


import random

class Neuron:
    def __init__(self, nin):
        # nin = number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        # x: list of floats or Values
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()   # or .relu()
        return out

    @property
    def params(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    @property
    def params(self):
        return [p for n in self.neurons for p in n.params]


class MLP:
    def __init__(self, nin, nouts):
        # nouts: e.g. [4, 4, 1]
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def params(self):
        return [p for layer in self.layers for p in layer.params]


# In[35]:


# toy data: 2D -> scalar
xs = [
    [2.0, 3.0],
    [1.0, -1.0],
    [-1.0, -2.0],
    [-2.0, 2.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # targets

model = MLP(2, [4, 4, 1])

for k in range(100):
    # forward
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))

    # zero grads
    for p in model.params:
        p.grad = 0.0

    # backward
    loss.backward()

    # SGD step
    lr = 0.05
    for p in model.params:
        p.data += -lr * p.grad

    if k % 10 == 0:
        print(k, loss.data)


# In[36]:


import numpy as np
import matplotlib.pyplot as plt

# ---- decision boundary plotting ----

# convert toy data to numpy for plotting
X_np = np.array(xs)
y_np = np.array(ys)

# grid limits (pad the data range a bit)
x1_min, x1_max = X_np[:, 0].min() - 1.0, X_np[:, 0].max() + 1.0
x2_min, x2_max = X_np[:, 1].min() - 1.0, X_np[:, 1].max() + 1.0

# meshgrid
xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 200),
    np.linspace(x2_min, x2_max, 200),
)

# evaluate model on grid
Z = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x1 = xx[i, j]
        x2 = yy[i, j]
        # model takes a list [x1, x2]; it will wrap scalars in Value internally
        out = model([x1, x2])
        # out is a Value
        Z[i, j] = out.data

# plot
plt.figure(figsize=(6, 6))

# filled contour for prediction values
# decision boundary roughly where model output == 0
cs = plt.contourf(
    xx, yy, Z,
    levels=50,
    alpha=0.6,
)

# decision boundary line (output = 0)
plt.contour(
    xx, yy, Z,
    levels=[0.0],
    colors="k",
    linewidths=2,
)

# scatter training points
# y = +1 one color, y = -1 another
pos = y_np > 0
neg = ~pos
plt.scatter(X_np[pos, 0], X_np[pos, 1], marker="o", edgecolors="k", label="y=+1")
plt.scatter(X_np[neg, 0], X_np[neg, 1], marker="x", edgecolors="k", label="y=-1")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision boundary of tiny MLP")
plt.legend()
plt.tight_layout()
plt.show()


# # Extending Micrograd to tensors

# In[37]:


class Value:
    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __matmul__(self, other):
        """
        Matrix multiply: self @ other

        Examples:
          self.data:  (B, D_in)
          other.data: (D_in, D_out)
          out.data:   (B, D_out)
        """
        out = Value(self.data @ other.data, _parents=(self, other), op="@")

        def _backward():
            """

            dL/dA (self grad)
            dL/dB (other grad)

            local grad = dY / dA and dY/dB

            incoming grad = out.grad

            dY / dA = B
            dY / dB = A

            dL / dA = out.grad @ BT
            dL / dB = AT @ out.grad

            """



            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad

        out._backward = _backward

        return out


# In[38]:


a = Value(3.0)
b = Value([1, 2, 3])
c = Value(np.random.randn(4, 5))

print(a.data.shape, a.grad.shape)
print(b.data.shape, b.grad.shape)
print(c.data.shape, c.grad.shape)


# In[39]:


A = Value([[1, 2, 3],
           [4, 5, 6]])

B = Value([[1, 0],
           [0, 1],
           [1, 1]])

C = A @ B

# EXPECTED SHAPE: (2, 2)
print(C.data.shape)


# In[40]:


X = Value(np.random.randn(5, 4))
W = Value(np.random.randn(4, 3))
Y = X @ W

# EXPECTED SHAPE: (5, 3)
print(Y.data.shape)


# In[ ]:





# In[41]:


class Value:
    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __matmul__(self, other):
        """
        Matrix multiply: self @ other

        Examples:
          self.data:  (B, D_in)
          other.data: (D_in, D_out)
          out.data:   (B, D_out)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, _parents=(self, other), op="@")

        def _backward():
            """

            dL/dA (self grad)
            dL/dB (other grad)

            local grad = dY / dA and dY/dB

            incoming grad = out.grad

            dY / dA = B
            dY / dB = A

            dL / dA = out.grad @ BT
            dL / dB = AT @ out.grad

            """



            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def sum(self):
        out = Value(np.sum(self.data), _parents=(self,), op="sum")

        def _backward():
            """
            small example, remember that..
                sum([x1, x2, x3, x4]) = x1+x2+x3+x4 = y

                dY/dx1 = 1
                dY/dx2 = 1
                dY/dx3 = 1
                dY/dx4 = 1

                addition gate just sends gradients downstream so...

                dL/dxi = dY/dXi * dL/dy

                returns a vector of 1 x dL/dy of shape (N, ) where N = len(self.data)
            """

            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """
        Backpropagate gradients from this Value all the way through
        the graph of Values that produced it.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # df/df = 1
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()


# In[42]:


x = Value([[1.0, 2.0],
           [3.0, 4.0]])
s = x.sum()
s.backward()

print(s.data)   # should be 10
print(x.grad)   # should be all ones with same shape as x.data


# In[43]:


math.tanh(np.array(3.0))


# In[44]:


class Value:
    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __matmul__(self, other):
        """
        Matrix multiply: self @ other

        Examples:
          self.data:  (B, D_in)
          other.data: (D_in, D_out)
          out.data:   (B, D_out)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, _parents=(self, other), op="@")

        def _backward():
            """

            dL/dA (self grad)
            dL/dB (other grad)

            local grad = dY / dA and dY/dB

            incoming grad = out.grad

            dY / dA = B
            dY / dB = A

            dL / dA = out.grad @ BT
            dL / dB = AT @ out.grad

            """



            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def sum(self):
        out = Value(np.sum(self.data), _parents=(self,), op="sum")

        def _backward():
            """
            small example, remember that..
                sum([x1, x2, x3, x4]) = x1+x2+x3+x4 = y

                dY/dx1 = 1
                dY/dx2 = 1
                dY/dx3 = 1
                dY/dx4 = 1

                addition gate just sends gradients downstream so...

                dL/dxi = dY/dXi * dL/dy

                returns a vector of 1 x dL/dy of shape (N, ) where N = len(self.data)
            """

            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    # These do not require backwards, as they are composed of the above primitives

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)    

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, _parents=(self,), op="**")

        def _backward():
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, _parents=(self, other), op='/')

        def _backward():
            """
            Remember that...
                f = x / y
                df/dx = 1 / y
                df/dy = -x / y^2
            """

            self.grad += 1.0 / (other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        # other / self
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def backward(self):
        """
        Backpropagate gradients from this Value all the way through
        the graph of Values that produced it.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # df/df = 1
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, _parents = (self,), op="tanh")

        def _backward():

            """
            Remember that...
                local gradient = dtanh(x)/dx = 1 - tanh(x)^2
            """
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out_mask = np.maximum(0, self.data)
        out = Value(out_mask, _parents=(self,), op="relu")

        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out


# In[45]:


x = Value(np.array([-1.0, 0.0, 2.0]))
y = x.relu()
s = y.sum()
s.backward()
print("y:", y.data)       # expect [0., 0., 2.]
print("x.grad:", x.grad)  # expect [0., 0., 1.]



# # Attempting Softmax

# In[46]:


np.exp([1,2,3])


# In[47]:


class Value:
    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __matmul__(self, other):
        """
        Matrix multiply: self @ other

        Examples:
          self.data:  (B, D_in)
          other.data: (D_in, D_out)
          out.data:   (B, D_out)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, _parents=(self, other), op="@")

        def _backward():
            """

            dL/dA (self grad)
            dL/dB (other grad)

            local grad = dY / dA and dY/dB

            incoming grad = out.grad

            dY / dA = B
            dY / dB = A

            dL / dA = out.grad @ BT
            dL / dB = AT @ out.grad

            """



            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims = False):
        out = Value(np.sum(self.data, axis=axis, keepdims=keepdims), _parents=(self,), op="sum")

        def _backward():
            """
            small example, remember that..
                sum([x1, x2, x3, x4]) = x1+x2+x3+x4 = y

                dY/dx1 = 1
                dY/dx2 = 1
                dY/dx3 = 1
                dY/dx4 = 1

                addition gate just sends gradients downstream so...

                dL/dxi = dY/dXi * dL/dy

                returns a vector of 1 x dL/dy of shape (N, ) where N = len(self.data)
            """

            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data =  self.data + other.data
        out = Value(data=child_data, _parents=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    # These do not require backwards, as they are composed of the above primitives

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)    

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, _parents=(self,), op="**")

        def _backward():
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, _parents=(self, other), op='/')

        def _backward():
            # x / y
            # dL/dx = (1 / y) * dL/dout
            self.grad += (1.0 / other.data) * out.grad

            # dL/dy = (-x / y^2) * dL/dout
            grad_other = (-self.data / (other.data ** 2)) * out.grad

            # if broadcasting happened (like (B, C) / (B, 1)),
            # we need to collapse along the broadcasted axis
            # so grad_other matches other.grad's shape
            while grad_other.ndim > other.data.ndim:
                grad_other = grad_other.sum(axis=0)  # rare case, but safe

            for axis, (s_o, s_g) in enumerate(zip(other.data.shape, grad_other.shape)):
                if s_o == 1 and s_g > 1:
                    grad_other = grad_other.sum(axis=axis, keepdims=True)

            other.grad += grad_other

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        # other / self
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def backward(self):
        """
        Backpropagate gradients from this Value all the way through
        the graph of Values that produced it.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # df/df = 1
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, _parents = (self,), op="tanh")

        def _backward():

            """
            Remember that...
                local gradient = dtanh(x)/dx = 1 - tanh(x)^2
            """
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out_mask = np.maximum(0, self.data)
        out = Value(out_mask, _parents=(self,), op="relu")

        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out
    def exp(self):
        """
        Elementwise exponential

        out = exp(self)
        """

        out = Value(np.exp(self.data), _parents=(self, ), op="exp")

        def _backward():
            """
            dL/dY = out.grad
            dL/dX = dY/dX * dL/dY
            dL/dX = e^x * dL/dY

            """
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return str(self.data)


# In[48]:


x = Value(np.arange(6).reshape(2, 3))
x


# In[49]:


s1 = x.sum()
s2 = x.sum(axis=1)
s3 = x.sum(axis=1, keepdims=True)


# In[50]:


s3


# In[51]:


def softmax(logits: Value, axis=-1):
    """
    logits: Value with shape (B, C)
    returns: Value with shape (B, C), row-wise softmax over classes
    """

    exps = logits.exp()
    sum = exps.sum(axis=axis, keepdims=True)
    return exps / sum


# In[52]:


logits = Value([[0.1, 0.003, 0.04, 0.05]])


# In[53]:


logits


# In[54]:


probs = softmax(logits)


# In[55]:


type(probs)


# In[56]:


logits


# In[57]:


logits = Value(np.random.randn(2, 4))
probs = softmax(logits)      # your softmax
s = probs.sum()              # scalar
s.backward()

print("probs:", probs.data)
print("logits.grad:", logits.grad)  # should be finite, no crash


# ## Attempting Cross entropy 

# In[58]:


class Value:
    def __init__(self, data, _parents=(), op="", label=None):
        """
        Initialize a scalar Value node for autograd.
        """
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.op = op
        self.label = label

        self._prev = set(_parents)
        self._backward = lambda: None

    def __matmul__(self, other):
        """
        Matrix multiply: self @ other

        Examples:
          self.data:  (B, D_in)
          other.data: (D_in, D_out)
          out.data:   (B, D_out)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, _parents=(self, other), op="@")

        def _backward():
            """

            dL/dA (self grad)
            dL/dB (other grad)

            local grad = dY / dA and dY/dB

            incoming grad = out.grad

            dY / dA = B
            dY / dB = A

            dL / dA = out.grad @ BT
            dL / dB = AT @ out.grad

            """



            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out


    def sum(self, axis=None, keepdims=False):
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Value(out_data, _parents=(self,), op="sum")

        def _backward():
            # upstream gradient as ndarray
            g = np.array(out.grad, dtype=float)

            # If we reduced over an axis and did NOT keep dims,
            # we need to reinsert that axis as size 1 so broadcasting works.
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)  # e.g. (4,) -> (4,1)

            # Now broadcast to input shape
            g_broadcast = np.broadcast_to(g, self.data.shape)

            # Accumulate
            self.grad = self.grad + g_broadcast

        out._backward = _backward
        return out


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data, _parents=(self, other), op='+')

        # def _backward():
        #     self.grad += out.grad
        #     other.grad += out.grad

        def _backward():
            grad_self = out.grad
            grad_other = out.grad

            self.grad += self._unbroadcast(grad_self, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        child_data = self.data * other.data
        out = Value(data=child_data, _parents=(self, other), op='*')

        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad

            self.grad += self._unbroadcast(grad_self, self.data.shape)
            other.grad += self._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    # These do not require backwards, as they are composed of the above primitives

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)    

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, _parents=(self,), op="**")

        def _backward():
            self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, _parents=(self, other), op='/')

        def _backward():
            # x / y
            grad_self  = (1.0 / other.data) * out.grad
            grad_other = (-self.data / (other.data ** 2)) * out.grad

            self.grad  += self._unbroadcast(grad_self,  self.data.shape)
            other.grad += self._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        # other / self
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def backward(self):
        """
        Backpropagate gradients from this Value all the way through
        the graph of Values that produced it.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # df/df = 1
        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, _parents = (self,), op="tanh")

        def _backward():

            """
            Remember that...
                local gradient = dtanh(x)/dx = 1 - tanh(x)^2
            """
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out_mask = np.maximum(0, self.data)
        out = Value(out_mask, _parents=(self,), op="relu")

        def _backward():
            self.grad += (self.data > 0).astype(float) * out.grad

        out._backward = _backward
        return out
    def exp(self):
        """
        Elementwise exponential

        out = exp(self)
        """

        out = Value(np.exp(self.data), _parents=(self, ), op="exp")

        def _backward():
            """
            dL/dY = out.grad
            dL/dX = dY/dX * dL/dY
            dL/dX = e^x * dL/dY

            """
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def T(self):
        return Value(self.data.T)

    def log(self):
        """
        Elementwise natural logarithm.

        out = log(self)
        """
        out = Value(np.log(self.data), _parents=(self,), op="log")

        def _backward():
            """
            dlog(x)/dx = 1/x
            dL/dx = out.grad * 1/x
            """
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out

    def mean(self, axis=None, keepdims=False):
        """
        Mean over all elements (axis=None) or over a given axis.
        """
        # 1) use your own sum(...)
        s = self.sum(axis=axis, keepdims=keepdims)

        # 2) figure out how many elements were summed
        if axis is None:
            count = self.data.size
        else:
            count = self.data.shape[axis]

        # 3) divide by count (your __truediv__ handles this)
        return s / count

    @staticmethod
    def _unbroadcast(grad, target_shape):
        """
        Reduce grad so that its shape matches target_shape, by summing over
        broadcasted dimensions.
        """
        g = grad

        # 1) Remove extra leading dims (if any)
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)

        # 2) For each axis, if target size is 1 but grad size > 1,
        #    that axis was broadcast → sum over it.
        for axis, (gdim, tdim) in enumerate(zip(g.shape, target_shape)):
            if tdim == 1 and gdim > 1:
                g = g.sum(axis=axis, keepdims=True)

        return g

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return str(self.data)


# In[59]:


def cross_entropy(logits: Value, targets_one_hot: Value):
    """
    logits: (B, C)
    targets_one_hot: (B, C), rows sum to 1
    returns: scalar Value (mean CE over batch)
    """

    probs = softmax(logits)                 # (B, C)
    log_probs = probs.log()                 # (B, C)
    loss_per_row = -(targets_one_hot * log_probs).sum(axis=1)   # (B,)

    loss_batch_average = loss_per_row.mean()

    return loss_batch_average



# In[60]:


logits = Value([[0.02, 0.03, 0.004]])
targets_one_hot = Value([[1,0,0]])


# In[61]:


CE = cross_entropy(logits, targets_one_hot)
CE


# In[62]:


logits = Value(np.random.randn(4, 3))
targets = np.eye(3)[np.array([0, 1, 2, 1])]
targets = Value(targets)

loss = cross_entropy(logits, targets)
loss.backward()

print("loss:", loss.data)
print("logits.grad.shape:", logits.grad.shape)  # should be (4, 3)


# In[63]:


np.array([1,2,3]) * - np.array(1)


# In[64]:


np.ones_like([[0,0,0,0], [0,0,0,0]]) * np.array([1,2,3,4])


# In[65]:


np.array([1,2,3,4]).shape


# # Building MNIST with our micro-tiny-grad

# In[66]:


class Linear:
    def __init__(self, in_features, out_features, activation = 'relu'):
        # W: (in_features, out_features)
        self.W = Value(np.random.randn(in_features, out_features) * 0.01)
        # b: (1, out_features)
        self.b = Value(np.zeros((1, out_features)))
        self.activation = activation

    def __call__(self, x: Value) -> Value:
        # x: (B, in_features)
        out = x @ self.W + self.b      # (B, out_features)
        # apply activation if requested
        if self.activation == "relu":
            out = out.relu()
        elif self.activation == "tanh":
            out = out.tanh()

        return out

    def parameters(self):
        return [self.W, self.b]


# In[67]:


class Network:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x: Value) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params


# In[ ]:





# In[68]:


import numpy as np
import tensorflow as tf

def load_mnist():
    """
    Load and preprocess MNIST dataset.

    Returns:
        X_train: Training images, shape (60000, 784), normalized to [0, 1]
        y_train: Training labels, shape (60000, 10), one-hot encoded
        X_test: Test images, shape (10000, 784), normalized to [0, 1]
        y_test: Test labels, shape (10000, 10), one-hot encoded
    """
    # Load MNIST from TensorFlow datasets
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Convert to numpy arrays (in case they aren't already)
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Flatten images from (28, 28) to (784,)
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # One-hot encode labels
    y_train = one_hot_encode(y_train, num_classes=10)
    y_test = one_hot_encode(y_test, num_classes=10)

    print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}, Labels: {y_test.shape}")

    return X_train, y_train, X_test, y_test

def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.

    Args:
        labels: Array of integer labels, shape (n_samples,)
        num_classes: Number of classes (default 10 for MNIST)

    Returns:
        One-hot encoded labels, shape (n_samples, num_classes)
    """
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot


# In[69]:


X_train, y_train, X_test, y_test = load_mnist()


# In[70]:


from tqdm import tqdm


# In[71]:


model = Network([
    Linear(784, 128, activation="relu"),
    Linear(128, 64, activation="relu"),
    Linear(64, 10, activation=None),   # logits layer
])
# X_train = X_train[0:100]
batch_size = 1000
lr = 0.10
n_samples = X_train.shape[0]
n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

losses = []

for epoch in range(10):
    permutation = np.random.permutation(n_samples)
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]
    epoch_loss = 0
    for batch_idx in tqdm(range(n_batches)):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        X_train_batch = X_train_shuffled[start:end]
        y_train_batch = y_train_shuffled[start:end]

        X_train_batch = Value(X_train_batch)
        y_train_batch = Value(y_train_batch)

        y_logits = model(X_train_batch)

        loss = cross_entropy(y_logits, y_train_batch)

        losses.append({'loss': float(loss.data)})
        loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad
            p.grad = np.zeros_like(p.grad)


# In[76]:


import pandas as pd
l = pd.DataFrame(losses)
l


# In[77]:


def add_smooth(df, col="loss", alpha=0.1):
    df = df.copy()
    df[col + "_smooth"] = df[col].ewm(alpha=alpha).mean()
    return df
l = add_smooth(l, col='loss')


# In[78]:


plt.figure(figsize=(12,6))
plt.plot(l["loss_smooth"], label="Smoothed", linewidth=2)
plt.plot(l["loss"], label="Raw", linewidth=2)


# In[79]:


pip install graphviz


# In[2]:


from graphviz import Digraph
import numpy as np

def trace(root):
    """
    Build the set of all nodes and edges in the graph
    leading to 'root'.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in v._prev:
                edges.add((parent, v))
                build(parent)

    build(root)
    return nodes, edges
def draw_dot(root, rankdir="LR"):
    """
    Visualize the computational graph from `root` backwards.

    Returns a graphviz.Digraph object. In Jupyter, just put it
    as the last line in a cell to display.
    """
    nodes, edges = trace(root)
    dot = Digraph(format="png",
                  graph_attr={"rankdir": rankdir})  # LR = left-to-right

    # Create a node for each Value
    for v in nodes:
        node_id = str(id(v))

        # Build a nice label
        label_lines = []
        if v.label is not None:
            label_lines.append(str(v.label))
        label_lines.append(f"shape={v.data.shape}")
        # small arrays: show values; big ones: skip
        if v.data.size <= 6:
            label_lines.append(f"value={np.round(v.data, 3)}")
        label = "\n".join(label_lines)

        dot.node(node_id, label=label, shape="record")

        # If this Value is the result of an op, create a separate op node
        if v.op:
            op_id = node_id + v.op
            dot.node(op_id, label=v.op, shape="circle")
            dot.edge(op_id, node_id)  # op -> value

    # Edges from parents to op nodes
    for parent, child in edges:
        parent_id = str(id(parent))
        child_id = str(id(child))

        if child.op:
            op_id = child_id + child.op
            dot.edge(parent_id, op_id)

    return dot


# In[12]:


# simple test
x = Value(np.array([[1.0, 2.0]]), label="x")
W = Value(np.random.randn(2, 3), label="W")
b = Value(np.zeros((1, 3)), label="b")

y = (x @ W + b).relu()
loss = y.sum()
loss.label = "loss"

dot = draw_dot(loss)
dot   # Jupyter will render it


# In[ ]:





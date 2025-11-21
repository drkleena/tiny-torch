import numpy as np

class Value:
    """
    Value is a node in the computation graph for autograd. It holds a numpy
    array and knows about its parents (other Values that were used to compute it).
    It also knows how to compute the gradient w.r.t. its parents.
    """

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
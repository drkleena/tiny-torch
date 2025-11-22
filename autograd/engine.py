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
        
        def _backward():
            grad_self = out.grad
            grad_other = out.grad

            self.grad += self._unbroadcast(grad_self, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
            
        out._backward = _backward
        return out
    
    __radd__ = __add__

    @staticmethod
    def _to_value(x):
        return x if isinstance(x, Value) else Value(x)

    def __gt__(self, other):
        """
        Elementwise greater-than comparison.
        Returns a Value whose data is a float mask of 0.0/1.0.
        Gradient does NOT flow through this (treated as constant mask).
        """
        other = self._to_value(other)
        mask = (self.data > other.data).astype(float)
        # no parents: mask is constant w.r.t. autograd
        return Value(mask)

    def __lt__(self, other):
        """
        Elementwise less-than comparison.
        Returns a Value whose data is a float mask of 0.0/1.0.
        """
        other = self._to_value(other)
        mask = (self.data < other.data).astype(float)
        return Value(mask)

    def clip(self, min_val, max_val):
        """
        Elementwise clip: min(max(self, min_val), max_val)
        Uses comparison masks; gradients flow only where not clipped.
        """

        # Make min / max broadcastable Values
        if not isinstance(min_val, Value):
            min_val = Value(np.full_like(self.data, float(min_val)))
        if not isinstance(max_val, Value):
            max_val = Value(np.full_like(self.data, float(max_val)))

        # Step 1: lower bound -> max(self, min_val)
        # mask_hi = 1 where self > min_val, else 0
        mask_hi = self > min_val                     # Value mask
        one_hi = Value(np.ones_like(mask_hi.data))   # 1s same shape
        x_hi = mask_hi * self + (one_hi - mask_hi) * min_val

        # Step 2: upper bound -> min(x_hi, max_val)
        # mask_lo = 1 where x_hi < max_val, else 0
        mask_lo = x_hi < max_val
        one_lo = Value(np.ones_like(mask_lo.data))
        x_clipped = mask_lo * x_hi + (one_lo - mask_lo) * max_val

        return x_clipped

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
    
    # def mean(self, axis=None, keepdims=False):
    #     """
    #     Mean over all elements (axis=None) or over a given axis.
    #     """
    #     # 1) use your own sum(...)
    #     s = self.sum(axis=axis, keepdims=keepdims)
    
    #     # 2) figure out how many elements were summed
    #     if axis is None:
    #         count = self.data.size
    #     else:
    #         count = self.data.shape[axis]
    
    #     # 3) divide by count (your __truediv__ handles this)
    #     return s / count
    """
    attempting to fix for case of autoencoder
    """
    import numpy as np

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)

        # how many elements were summed?
        if axis is None:
            count = self.data.size
        else:
            if isinstance(axis, int):
                axis = (axis,)
            # count elements along all reduced axes
            count = np.prod([self.data.shape[a] for a in axis])

        return s / count

    def transpose(self, axes=None):
        """
        Permute the dimensions of this Value's data.

        axes: tuple of ints specifying the new order of axes.
              If None, reverse the axes order (like numpy).
        """
        if axes is None:
            axes_final = tuple(reversed(range(self.data.ndim)))
        else:
            axes_final = tuple(axes)

        out_data = np.transpose(self.data, axes_final)
        out = Value(out_data, _parents=(self,), op="transpose")

        def _backward():
            # inverse permutation: if axes = (2,0,1), inv_axes = (1,2,0)
            inv_axes = np.argsort(axes_final)
            self.grad += np.transpose(out.grad, tuple(inv_axes))
        
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out = Value(out_data, _parents=(self,), op="getitem")

        def _backward():
            g_full = np.zeros_like(self.data)
            # Proper scatter-add for advanced indexing
            """
            np.add.at applies additive updates per index occurrence,
            so if a location appears multiple times in idx, it gets incremented multiple times.
            """
            np.add.at(g_full, idx, out.grad)
            self.grad += g_full

        out._backward = _backward
        return out

    @staticmethod
    def stack(values, axis=0):
        datas = [v.data for v in values]
        out_data = np.stack(datas, axis=axis)
        out = Value(out_data, _parents=tuple(values), op="stack")

        def _backward():
            # incoming grad: shape (B, C, H, W)
            grads = np.split(out.grad, len(values), axis=axis)
            grads = [g.squeeze(axis=axis) for g in grads]
            for v, g in zip(values, grads):
                v.grad += g

        out._backward = _backward
        return out

    @staticmethod
    def concat(values, axis=0):
        """
        Concatenate a list of Value tensors along a given axis.
        All non-concat dimensions must match (like np.concatenate).
        """
        datas = [v.data for v in values]
        out_data = np.concatenate(datas, axis=axis)

        out = Value(out_data, _parents=tuple(values), op="concat")

        def _backward():
            # sizes along the concat axis for each input
            sizes = [v.data.shape[axis] for v in values]
            # boundaries to split out.grad
            offsets = np.cumsum(sizes)[:-1]

            # split incoming gradient along the same axis
            grads = np.split(out.grad, offsets, axis=axis)

            # route each chunk to its corresponding parent
            for v, g in zip(values, grads):
                v.grad += g

        out._backward = _backward

        return out

    def reshape(self, new_shape):
        out_data = np.reshape(self.data, new_shape)
        out = Value(out_data, _parents=(self,), op="reshape")

        def _backward():
            self.grad += np.reshape(out.grad, self.data.shape)

        out._backward = _backward
        return out

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
    

    def pad(self, pad_h: int, pad_w: int):
        """
        Zero-pad a (C, H, W) tensor along H and W.
        pad_h, pad_w: number of zeros on both sides (top/bottom, left/right).
        """

        C, H, W = self.data.shape

        pad_width = (
            (0, 0),          # C
            (pad_h, pad_h),  # H
            (pad_w, pad_w),  # W
        )

        out_data = np.pad(self.data, pad_width, mode='constant', constant_values=0.0)
        out = Value(out_data, _parents=(self,), op="pad")        
            
        def _backward():
            if pad_h == 0 and pad_w == 0:
                # just send the grads on their way back
                self.grad += out.grad
            else:
                # unpad the guy
                grad_unpad = out.grad[:, pad_h: H + pad_h, pad_w: W + pad_w]
                self.grad += grad_unpad
        out._backward = _backward
        return out

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return str(self.data)
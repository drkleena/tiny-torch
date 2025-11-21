import numpy as np

class Adam:
    """
    Adam optimizer
    - lr:       learning rate (default 1e-3)
    - betas:    (beta_1, beta_2) as in Adam paper (defaults (0.9, 0.999))
    - eps:      epsilon for numerical stability (default 1e-7, like Keras)
    - weight_decay: L2 weight decay (0.0 = off, Keras default)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # iteration counter (starts at 0, incremented before each step)
        self.t = 0

        # per-parameter state
        # use dict keyed by id(param) so we don't depend on list order
        self.m = {}
        self.v = {}
        for p in self.params:
            self.m[id(p)] = np.zeros_like(p.data, dtype=p.data.dtype)
            self.v[id(p)] = np.zeros_like(p.data, dtype=p.data.dtype)

    def step(self):
        """
        Perform a single optimization step.
        """
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr, eps = self.lr, self.eps

        for p in self.params:
            g = p.grad
            if g is None:
                continue

            # Optional: L2 weight decay (Keras Adam defaults to no decay)
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            pid = id(p)
            m = self.m[pid]
            v = self.v[pid]

            # First moment
            m = b1 * m + (1.0 - b1) * g
            # Second moment
            v = b2 * v + (1.0 - b2) * (g * g)

            # Bias correction
            m_hat = m / (1.0 - b1 ** self.t)
            v_hat = v / (1.0 - b2 ** self.t)

            # Parameter update (matches Keras: lr * m_hat / (sqrt(v_hat) + eps))
            p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)

            # Store updated moments
            self.m[pid] = m
            self.v[pid] = v

    def zero_grad(self):
        """
        Reset gradients to zero for all parameters.
        """
        for p in self.params:
            if p.grad is not None:
                # ensure we keep the same shape & dtype
                p.grad[...] = 0.0

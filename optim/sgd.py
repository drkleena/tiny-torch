"""
Stochastic Gradient Descent optimizer.
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Performs parameter updates using gradient descent:
        param = param - learning_rate * gradient

    Args:
        params: Iterable of Value objects representing model parameters
        lr: Learning rate (step size) for parameter updates

    Example:
        >>> from nn import Network, Linear
        >>> model = Network([Linear(10, 5), Linear(5, 1)])
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>>
        >>> # Training loop
        >>> loss = compute_loss(...)
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """

    def __init__(self, params, lr=0.1):
        """
        Initialize the SGD optimizer.

        Args:
            params: List of Value objects (model parameters) to optimize
            lr: Learning rate (default: 0.1)
        """
        self.params = list(params)
        self.lr = lr

    def step(self):
        """
        Perform a single optimization step (parameter update).

        Updates each parameter by subtracting lr * gradient from its current value.
        Call this after backward() has computed gradients.
        """
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Reset all parameter gradients to zero.

        Call this before each backward pass to clear gradients from the previous step.
        """
        for p in self.params:
            p.grad = np.zeros_like(p.grad)

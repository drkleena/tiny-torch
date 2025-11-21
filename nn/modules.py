"""
Neural network modules and layers.

This module contains stateful layer classes like Linear, which maintain
parameters (weights and biases).
"""

import numpy as np
from autograd.engine import Value


class Linear:
    """
    Fully connected (dense) linear layer with optional activation.

    Performs the operation: out = x @ W + b
    Optionally applies an activation function (relu or tanh).

    Args:
        in_features: Number of input features
        out_features: Number of output features
        activation: Optional activation function ('relu', 'tanh', or None)

    Example:
        >>> layer = Linear(784, 128, activation='relu')
        >>> x = Value(np.random.randn(32, 784))  # Batch of 32
        >>> output = layer(x)  # Shape: (32, 128)
    """
    
    def __init__(self, in_features, out_features, activation=None):
        # W: (in_features, out_features)
        std = np.sqrt(2.0 / in_features)
        self.W = Value(np.random.randn(in_features, out_features) * std)
        # b: (1, out_features)
        self.b = Value(np.zeros((1, out_features)))
        self.activation = activation

    def __call__(self, x: Value) -> Value:
        """
        Forward pass through the layer.

        Args:
            x: Input Value with shape (B, in_features)

        Returns:
            Output Value with shape (B, out_features)
        """
        # x: (B, in_features)
        out = x @ self.W + self.b  # (B, out_features)

        # Apply activation if requested
        if self.activation == "relu":
            out = out.relu()
        elif self.activation == "tanh":
            out = out.tanh()

        return out

    def parameters(self):
        """Return list of trainable parameters (weights and biases)."""
        return [self.W, self.b]


class Network:
    """
    Sequential neural network composed of multiple layers.

    Args:
        layers: List of layer modules (e.g., Linear instances)

    Example:
        >>> model = Network([
        ...     Linear(784, 128, activation='relu'),
        ...     Linear(128, 64, activation='relu'),
        ...     Linear(64, 10, activation=None)  # Logits layer
        ... ])
        >>> x = Value(np.random.randn(32, 784))
        >>> logits = model(x)  # Shape: (32, 10)
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x: Value) -> Value:
        """
        Forward pass through all layers in sequence.

        Args:
            x: Input Value

        Returns:
            Output Value after passing through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Collect all trainable parameters from all layers.

        Returns:
            List of all Value objects representing trainable parameters
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

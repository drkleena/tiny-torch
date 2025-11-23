"""
Neural network modules and layers.

This module contains stateful layer classes like Linear, which maintain
parameters (weights and biases).
"""

import numpy as np
from autograd.engine import Value
from .functional import sigmoid, conv2d, max_pool2d

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
        # Using He initialization for better training with ReLU
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
        elif self.activation == "sigmoid":
            out = sigmoid(out)

        return out

    def parameters(self):
        """Return list of trainable parameters (weights and biases)."""
        return [self.W, self.b]

class Conv2D:
    """
    2D convolutional layer with optional activation.

    Performs the operation: out = x @ W + b
    Optionally applies an activation function (relu or tanh).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel (height, width)
        stride: Stride of the convolution
        padding: Padding of the convolution
        activation: Optional activation function ('relu', 'tanh', or None)

    Example:
        >>> layer = Conv2D(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        >>> x = Value(np.random.randn(32, 1, 28, 28))  # Batch of 32
        >>> output = layer(x)  # Shape: (32, 32, 28, 28)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
            assert kH == kW, "for now we only support square kernels"
        
        self.in_channels = in_channels # input image channels
        self.out_channels = out_channels # output channels, or, number of filters
        self.kernel_size = kernel_size # kernel size (square)
        self.stride = stride # stride
        self.padding = padding # padding
        self.activation = activation # activation function

        # param init
        scale = np.sqrt(2.0 / (in_channels * kH * kW))
        W_np = np.random.randn(out_channels, in_channels, kH, kW) * scale
        b_np = np.zeros(out_channels)

        # auto gradable params
        self.W = Value(W_np)
        self.b = Value(b_np)

    def parameters(self):
        """Return list of trainable parameters (weights and biases)."""
        return [self.W, self.b]
    
    def __call__(self, x: Value) -> Value:
        out = conv2d(x, self.W, self.b, stride=self.stride, padding=self.padding)
        # Apply activation if requested
        if self.activation == "relu":
            out = out.relu()
        elif self.activation == "tanh":
            out = out.tanh()
        elif self.activation == "sigmoid":
            out = sigmoid(out)
        return out # add dummy batch dim ?

class Flatten:
    """
    Flatten (N, C, H, W) -> (N, C*H*W)
    or (C, H, W) -> (1, C*H*W)
    """
    def __call__(self, x: Value) -> Value:
        data = x.data
        if data.ndim == 4:
            N = data.shape[0]
            return x.reshape((N, -1))
        elif data.ndim == 3:
            # single sample (C,H,W)
            return x.reshape((1, -1))
        else:
            # already flat, do nothing
            return x

class MaxPool2D:
    """
    Max pooling layer.
    """
    def __init__(self, pool_size, stride=1):
        self.pool_size = pool_size
        self.stride = stride
    
    def __call__(self, x: Value) -> Value:
        return max_pool2d(x, self.pool_size, self.stride)


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

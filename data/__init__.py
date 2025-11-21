"""
Data loading utilities for tiny-micro-torch.

Provides functions for loading and preprocessing datasets like MNIST.
"""

from .mnist import load_mnist, one_hot_encode

__all__ = ['load_mnist', 'one_hot_encode']

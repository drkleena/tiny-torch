"""
MNIST dataset loading and preprocessing utilities.
"""

import numpy as np
import tensorflow as tf


def load_mnist():
    """
    Load and preprocess MNIST dataset.

    Downloads MNIST using TensorFlow, normalizes pixel values to [0, 1],
    flattens images from (28, 28) to (784,), and one-hot encodes labels.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            - X_train: Training images, shape (60000, 784), normalized to [0, 1]
            - y_train: Training labels, shape (60000, 10), one-hot encoded
            - X_test: Test images, shape (10000, 784), normalized to [0, 1]
            - y_test: Test labels, shape (10000, 10), one-hot encoded

    Example:
        >>> X_train, y_train, X_test, y_test = load_mnist()
        >>> print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
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

    Example:
        >>> labels = np.array([0, 2, 1, 1])
        >>> one_hot = one_hot_encode(labels, num_classes=3)
        >>> print(one_hot)
        [[1. 0. 0.]
         [0. 0. 1.]
         [0. 1. 0.]
         [0. 1. 0.]]
    """
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot

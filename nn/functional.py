"""
Functional operations for neural networks.

This module contains stateless operations like activation functions,
softmax, and loss functions.
"""

from autograd.engine import Value


def softmax(logits: Value, axis=-1):
    """
    Compute softmax activation over the specified axis.

    Args:
        logits: Value with shape (B, C) or any shape where softmax is computed along axis
        axis: Axis along which to compute softmax (default: -1, last axis)

    Returns:
        Value with same shape as logits, normalized probabilities

    Example:
        >>> logits = Value([[1.0, 2.0, 3.0]])
        >>> probs = softmax(logits)  # Row-wise softmax
    """
    exps = logits.exp()
    sum_exps = exps.sum(axis=axis, keepdims=True)
    return exps / sum_exps

def cross_entropy(logits: Value, targets_one_hot: Value):
    """
    Compute cross-entropy loss between logits and one-hot targets.

    Args:
        logits: Value with shape (B, C) - raw model outputs (before softmax)
        targets_one_hot: Value with shape (B, C) - one-hot encoded labels

    Returns:
        Scalar Value representing the mean cross-entropy loss over the batch

    Example:
        >>> logits = Value([[0.1, 0.2, 0.7]])
        >>> targets = Value([[0.0, 0.0, 1.0]])  # Class 2 is correct
        >>> loss = cross_entropy(logits, targets)
    """
    probs = softmax(logits)  # (B, C)
    log_probs = probs.log()  # (B, C)
    loss_per_row = -(targets_one_hot * log_probs).sum(axis=1)  # (B,)
    loss_batch_average = loss_per_row.mean()

    return loss_batch_average

def mse(predictions: Value, targets: Value):
    """
    Compute mean squared error between predictions and targets.

    Args:
        predictions: Value with shape (B, ...) - model predictions
        targets: Value with shape (B, ...) - ground truth values

    Returns:
        Scalar Value representing the mean squared error over the batch
    """
    squared_errors = (predictions - targets) ** 2
    return squared_errors.mean()

def sigmoid(x: Value):
    """
    Compute sigmoid activation.

    Args:
        x: Value to apply sigmoid to

    Returns:
        Value with sigmoid applied: 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + (-x).exp())

def binary_cross_entropy(pred: Value, target: Value, eps=1e-7) -> Value:
    pred_safe = pred.clip(eps, 1.0 - eps)
    term1 = target * pred_safe.log()
    term2 = (1 - target) * (1 - pred_safe).log()
    return -(term1 + term2).mean()

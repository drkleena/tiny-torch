"""
Functional operations for neural networks.

This module contains stateless operations like activation functions,
softmax, and loss functions.
"""

from autograd.engine import Value
import numpy as np

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

def im2col(X: Value, kernel_size, stride=1, padding=0):
    """
    X: Value with data.shape (N, C, H, W)
    padding: int, symmetric zero-padding on H and W
    """
    k = kernel_size

    # 1) optionally pad
    if padding > 0:
        Xps = Value.stack([ input.pad(padding, padding) for input in X ])   # (B, C, H+2p, W+2p)
    else:
        Xps = X

    B, C, xH, xW = Xps.data.shape
     
    # 2) output spatial dims
    steps_y = (xH - k) // stride + 1
    steps_x = (xW - k) // stride + 1

    # # 3) coordinate grid (numpy-only, fine)
    y_coords, x_coords = np.meshgrid(
        np.arange(steps_y),
        np.arange(steps_x),
        indexing='ij'
    )    
    
    y_idx = y_coords[:, :, None, None] * stride + np.arange(k)[None, None, :, None]
    x_idx = x_coords[:, :, None, None] * stride + np.arange(k)[None, None, None, :]

    patches = Xps[:, :, y_idx, x_idx] # (B, C, Sy, Sx, k, k)
    # 5) transpose to (Sy, Sx, C, k, k)
    patches_transpose = patches.transpose((0, 2, 3, 1, 4, 5))  # (B, Sy, Sx, C, k, k)
    B, Sy, Sx, C, k1, k2 = patches_transpose.data.shape
    
    num_patches = steps_y * steps_x        # Sy * Sx
    patch_area = k * k * C                 # C*k*k

    # 6) flatten patches → (num_patches, patch_area)
    patches_flat = patches_transpose.reshape((B, num_patches, patch_area))

    # 7) final cols → (patch_area, num_patches)
    cols = patches_flat.transpose((0, 2, 1))
    return cols

def im2patches(X: Value, kernel_size, stride=1, padding=0) -> Value:
    """
    Extracts patches from an image.

    Parameters
    ----------
    X : Value
        Input image with shape (N, C, H, W), where N is the batch size, C is the number of channels,
        H is the height, and W is the width.
    kernel_size : int or tuple of two ints
        Size of the kernel to be applied. If it is an `int`, the kernel will be square.
        Otherwise, it should be a tuple of two ints, where the first element is the height
        and the second element is the width of the kernel.
    stride : int, optional
        Step size between each patch. Default is 1.
    padding : int, optional
        Zero-padding added to the input image. Default is 0.

    Returns
    -------
    Value
        Patches extracted from the input image with shape (N, C*k*k, Sy*Sx), where N is the batch size,
        C is the number of channels, k is the size of the kernel, Sy is the number of rows of patches,
        and Sx is the number of columns of patches.
    """
    k = kernel_size

    # 1) optionally pad
    if padding > 0:
        Xps = Value.stack([ input.pad(padding, padding) for input in X ])   # (B, C, H+2p, W+2p)
    else:
        Xps = X

    B, C, xH, xW = Xps.data.shape
     
    # 2) output spatial dims
    steps_y = (xH - k) // stride + 1
    steps_x = (xW - k) // stride + 1

    # # 3) coordinate grid (numpy-only, fine)
    y_coords, x_coords = np.meshgrid(
        np.arange(steps_y),
        np.arange(steps_x),
        indexing='ij'
    )    
    
    y_idx = y_coords[:, :, None, None] * stride + np.arange(k)[None, None, :, None]
    x_idx = x_coords[:, :, None, None] * stride + np.arange(k)[None, None, None, :]

    patches = Xps[:, :, y_idx, x_idx] # (B, C, Sy, Sx, k, k)
    return patches    

def max_pool2d(x: Value, pool_size, stride=1, padding=0):
    B, C, H, W = x.data.shape
    pool_h = pool_w = pool_size
    
    H_out = (H - pool_h) // 2
    W_out = (W - pool_w) // 2

    # X: Value, kernel_size, stride=1, padding=0
    patches = im2patches(x, pool_size, stride, padding)
    
    # patches shape : (2, 1, 2, 2, 2, 2) (B, C, Sy, Sx, k, k)
    
    B, C, Sy, Sx, k, _ = patches.data.shape
    patches_flat = patches.reshape((B, C, Sy, Sx, k * k))

    # Take max over the last dimension (all pixels in each patch)
    # This gives shape (B, C, Sy, Sx)
    max_vals = patches_flat.max(axis=-1)
    
    return max_vals

def conv2d(x: Value, weight: Value, bias: Value, stride: int = 1, padding: int = 0) -> Value:
    """
    x:       (B, C_in, H, W)
    weight:  (C_out, C_in, k, k)
    bias:    (C_out,)
    """
    B, C_in, H, W = x.data.shape
    C_out, C_in_w, kH, kW = weight.data.shape
    assert C_in == C_in_w
    assert kH == kW
    k = kH

    cols = im2col(x, kernel_size=k, stride=stride, padding=padding) # (B, patch_area, num_patches)
    Bc, patch_area, num_patches = cols.data.shape
    assert Bc == B
    assert patch_area == C_in * k * k

    cols_2d = cols.transpose((0, 2, 1)).reshape((B * num_patches, patch_area))

    W_mat = weight.reshape((C_out, patch_area)).transpose((1, 0))  # (patch_area, C_out)

    out_2d = cols_2d @ W_mat

    out_3d = out_2d.reshape((B, num_patches, C_out)).transpose((0, 2, 1))  # (B, C_out, num_patches)

    out_3d = out_3d + bias.reshape((1, C_out, 1))

    # reshape to (B, C_out, H_out, W_out)
    H_p = H + 2 * padding
    W_p = W + 2 * padding
    H_out = (H_p - k) // stride + 1
    W_out = (W_p - k) // stride + 1
    assert num_patches == H_out * W_out

    out = out_3d.reshape((B, C_out, H_out, W_out))
    return out
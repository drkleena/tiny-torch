
import numpy as np
from autograd.engine import Value
from nn.functional import (
    im2col,
    softmax,
    cross_entropy,
    mse,
    sigmoid,
    binary_cross_entropy,
    conv2d_single,
    im2patches,
    max_pool2d
)

def test_im2col_single_channel_3x3_k2_s1():
    X_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # shape (1, 1, 3, 3) - batch_size=1, channels=1

    X = Value(X_np)
    cols = im2col(X, kernel_size=2, stride=1)

    expected = np.array([[
        [1., 2., 4., 5.],
        [2., 3., 5., 6.],
        [4., 5., 7., 8.],
        [5., 6., 8., 9.],
    ]])  # shape (1, 4, 4) - batch_size=1

    assert cols.data.shape == expected.shape
    assert np.allclose(cols.data, expected)

def test_im2col_two_channels_3x3_k2_s1():
    X_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ],  # C0
        [
            [11., 12., 13.],
            [14., 15., 16.],
            [17., 18., 19.],
        ],  # C1
    ]])  # shape (1, 2, 3, 3) - batch_size=1, channels=2

    X = Value(X_np)
    cols = im2col(X, kernel_size=2, stride=1)

    # Expected: each column is [C0_patch_flat; C1_patch_flat]
    expected = np.array([[
        [ 1.,  2.,  4.,  5.],
        [ 2.,  3.,  5.,  6.],
        [ 4.,  5.,  7.,  8.],
        [ 5.,  6.,  8.,  9.],
        [11., 12., 14., 15.],
        [12., 13., 15., 16.],
        [14., 15., 17., 18.],
        [15., 16., 18., 19.],
    ]])  # shape (1, 8, 4) - batch_size=1

    assert cols.data.shape == expected.shape
    assert np.allclose(cols.data, expected)

def test_im2col_stride_shape():
    X_np = np.arange(1 * 1 * 5 * 5, dtype=float).reshape(1, 1, 5, 5)
    X = Value(X_np)

    k = 3

    cols_s1 = im2col(X, kernel_size=k, stride=1)
    cols_s2 = im2col(X, kernel_size=k, stride=2)

    # H_out = (H - k)//stride + 1
    H_out_s1 = (5 - k)//1 + 1  # 3
    H_out_s2 = (5 - k)//2 + 1  # 2

    # rows = C * k * k = 1 * 3 * 3 = 9
    assert cols_s1.data.shape == (1, 9, H_out_s1 * H_out_s1)  # (1, 9, 9)
    assert cols_s2.data.shape == (1, 9, H_out_s2 * H_out_s2)  # (1, 9, 4)

def test_im2col_backward_overlaps():
    X_np = np.arange(1 * 1 * 3 * 3, dtype=float).reshape(1, 1, 3, 3)
    X = Value(X_np)

    cols = im2col(X, kernel_size=2, stride=1)  # (1, 4, 4)
    loss = cols.sum()                          # scalar

    loss.backward()

    # Each pixel participation count:
    # corners: 1, edges (non-corner): 2, center: 4
    expected_counts = np.array([[
        [
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.],
        ]
    ]])

    assert X.grad.shape == expected_counts.shape
    assert np.allclose(X.grad, expected_counts)

def test_im2col_conv_equivalence():
    X_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # (1, 1, 3, 3)

    X = Value(X_np)

    k = 2
    cols = im2col(X, kernel_size=k, stride=1)   # (1, 4, 4)

    # Single filter, all ones → sums of each 2x2 patch
    W_np = np.ones((1, 1, k, k), dtype=float)
    W = Value(W_np)

    W_mat = W.reshape((1, 1 * k * k))          # (1, 4)
    Y_cols = W_mat @ cols                      # (1, 1, 4) - broadcasts over batch
    Y = Y_cols.reshape((1, 1, 2, 2))           # (1, 1, 2, 2)

    expected = np.array([[
        [
            [1+2+4+5, 2+3+5+6],
            [4+5+7+8, 5+6+8+9],
        ]
    ]])

    assert np.allclose(Y.data, expected)

def test_im2col_padding_forward():
    X_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # (1, 1, 3, 3)

    X = Value(X_np)
    cols = im2col(X, kernel_size=3, stride=1, padding=1)

    # padded input is (1, 1, 5, 5)
    # H_out = W_out = (5 - 3)//1 + 1 = 3
    # rows = C*k*k = 1*3*3 = 9
    assert cols.data.shape == (1, 9, 9)

def test_im2col_padding_backward():
    X_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # (1, 1, 3, 3)

    X = Value(X_np)

    # single padding stage: let im2col handle pad
    cols = im2col(X, kernel_size=3, stride=1, padding=1)
    loss = cols.sum()
    loss.backward()

    expected = np.array([[
        [
            [4., 6., 4.],
            [6., 9., 6.],
            [4., 6., 4.],
        ]
    ]])  # shape (1, 1, 3, 3) to match X.grad

    # or Option 2: exact equality
    assert (X.grad == expected).all()

def test_softmax_basic():
    logits_np = np.array([[1.0, 2.0, 3.0]])
    logits = Value(logits_np)

    probs = softmax(logits)

    # Expected: exp(logits) / sum(exp(logits))
    exp_vals = np.exp(logits_np)
    expected = exp_vals / exp_vals.sum(axis=-1, keepdims=True)

    assert probs.data.shape == expected.shape
    assert np.allclose(probs.data, expected)

def test_softmax_sum_to_one():
    logits_np = np.array([[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0]])
    logits = Value(logits_np)

    probs = softmax(logits)

    # Check that each row sums to 1
    sums = probs.data.sum(axis=-1)
    assert np.allclose(sums, np.ones(2))

def test_softmax_backward():
    logits_np = np.array([[1.0, 2.0, 3.0]])
    logits = Value(logits_np)

    probs = softmax(logits)
    loss = probs.sum()

    loss.backward()

    # Gradient should exist and have same shape as input
    assert logits.grad.shape == logits_np.shape
    assert logits.grad is not None

def test_cross_entropy_basic():
    logits_np = np.array([[0.1, 0.2, 0.7]])
    targets_np = np.array([[0.0, 0.0, 1.0]])  # Class 2 is correct

    logits = Value(logits_np)
    targets = Value(targets_np)

    loss = cross_entropy(logits, targets)

    # Expected: -log(softmax(logits)[target_class])
    probs_np = np.exp(logits_np) / np.exp(logits_np).sum(axis=-1, keepdims=True)
    expected_loss = -(targets_np * np.log(probs_np)).sum(axis=-1).mean()

    assert loss.data.shape == ()  # Scalar
    assert np.allclose(loss.data, expected_loss)

def test_cross_entropy_backward():
    logits_np = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
    targets_np = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])

    logits = Value(logits_np)
    targets = Value(targets_np)

    loss = cross_entropy(logits, targets)
    loss.backward()

    # Gradient should exist and have same shape as input
    assert logits.grad.shape == logits_np.shape
    assert logits.grad is not None

def test_cross_entropy_perfect_prediction():
    # Very high confidence in correct class
    logits_np = np.array([[0.0, 0.0, 10.0]])
    targets_np = np.array([[0.0, 0.0, 1.0]])

    logits = Value(logits_np)
    targets = Value(targets_np)

    loss = cross_entropy(logits, targets)

    # Loss should be very small for perfect prediction
    assert loss.data < 0.01

def test_mse_basic():
    predictions_np = np.array([[1.0, 2.0, 3.0]])
    targets_np = np.array([[1.5, 2.5, 3.5]])

    predictions = Value(predictions_np)
    targets = Value(targets_np)

    loss = mse(predictions, targets)

    # Expected: mean((predictions - targets)^2)
    expected = ((predictions_np - targets_np) ** 2).mean()

    assert loss.data.shape == ()  # Scalar
    assert np.allclose(loss.data, expected)

def test_mse_zero_loss():
    predictions_np = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])

    predictions = Value(predictions_np)
    targets = Value(predictions_np)  # Same as predictions

    loss = mse(predictions, targets)

    # Loss should be exactly zero
    assert np.allclose(loss.data, 0.0)

def test_mse_backward():
    predictions_np = np.array([[1.0, 2.0, 3.0]])
    targets_np = np.array([[2.0, 3.0, 4.0]])

    predictions = Value(predictions_np)
    targets = Value(targets_np)

    loss = mse(predictions, targets)
    loss.backward()

    # Gradient should exist and have same shape as predictions
    assert predictions.grad.shape == predictions_np.shape
    # Gradient should be 2*(predictions - targets) / n
    expected_grad = 2 * (predictions_np - targets_np) / predictions_np.size
    assert np.allclose(predictions.grad, expected_grad)

def test_sigmoid_basic():
    x_np = np.array([[0.0, 1.0, -1.0, 2.0]])
    x = Value(x_np)

    result = sigmoid(x)

    # Expected: 1 / (1 + exp(-x))
    expected = 1.0 / (1.0 + np.exp(-x_np))

    assert result.data.shape == expected.shape
    assert np.allclose(result.data, expected)

def test_sigmoid_bounds():
    x_np = np.array([[-100.0, 0.0, 100.0]])
    x = Value(x_np)

    result = sigmoid(x)

    # Sigmoid should be bounded between 0 and 1
    assert np.all(result.data >= 0.0)
    assert np.all(result.data <= 1.0)

    # At x=0, sigmoid should be 0.5
    assert np.allclose(result.data[0, 1], 0.5)

def test_sigmoid_backward():
    x_np = np.array([[1.0, 2.0, 3.0]])
    x = Value(x_np)

    result = sigmoid(x)
    loss = result.sum()

    loss.backward()

    # Gradient should exist and have same shape as input
    assert x.grad.shape == x_np.shape
    assert x.grad is not None

def test_binary_cross_entropy_basic():
    pred_np = np.array([[0.7, 0.3, 0.9]])
    target_np = np.array([[1.0, 0.0, 1.0]])

    pred = Value(pred_np)
    target = Value(target_np)

    loss = binary_cross_entropy(pred, target)

    # Manual calculation with epsilon clipping
    eps = 1e-7
    pred_safe = np.clip(pred_np, eps, 1.0 - eps)
    expected = -(target_np * np.log(pred_safe) + (1 - target_np) * np.log(1 - pred_safe)).mean()

    assert loss.data.shape == ()  # Scalar
    assert np.allclose(loss.data, expected)

def test_binary_cross_entropy_perfect_prediction():
    pred_np = np.array([[0.999999, 0.000001]])
    target_np = np.array([[1.0, 0.0]])

    pred = Value(pred_np)
    target = Value(target_np)

    loss = binary_cross_entropy(pred, target)

    # Loss should be very small for perfect prediction
    assert loss.data < 0.01

def test_binary_cross_entropy_backward():
    pred_np = np.array([[0.6, 0.4, 0.8]])
    target_np = np.array([[1.0, 0.0, 1.0]])

    pred = Value(pred_np)
    target = Value(target_np)

    loss = binary_cross_entropy(pred, target)
    loss.backward()

    # Gradient should exist
    assert pred.grad is not None
    assert pred.grad.shape == pred_np.shape

def test_conv2d_single_basic():
    # Single batch, single channel input (1, 1, 3, 3)
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])

    # Single output channel, single input channel (1, 1, 2, 2)
    weight_np = np.ones((1, 1, 2, 2))
    bias_np = np.array([0.0])

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=0)

    # Expected: sum of each 2x2 patch (1, 1, 2, 2)
    expected = np.array([[
        [
            [1+2+4+5, 2+3+5+6],
            [4+5+7+8, 5+6+8+9],
        ]
    ]])

    assert result.data.shape == expected.shape
    assert np.allclose(result.data, expected)

def test_conv2d_single_with_bias():
    # Single batch, single channel input (1, 1, 3, 3)
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])

    weight_np = np.ones((1, 1, 2, 2))
    bias_np = np.array([10.0])  # Non-zero bias

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=0)

    # Expected: sum of each 2x2 patch + 10 (1, 1, 2, 2)
    expected = np.array([[
        [
            [1+2+4+5+10, 2+3+5+6+10],
            [4+5+7+8+10, 5+6+8+9+10],
        ]
    ]])

    assert np.allclose(result.data, expected)

def test_conv2d_single_multiple_output_channels():
    # Single batch, single channel input (1, 1, 3, 3)
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])

    # Two output channels
    weight_np = np.ones((2, 1, 2, 2))
    weight_np[1] = 2.0  # Second channel has weight 2
    bias_np = np.array([0.0, 0.0])

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=0)

    # Shape should be (1, 2, 2, 2) - batch=1, 2 output channels
    assert result.data.shape == (1, 2, 2, 2)

    # First channel: same as basic test
    expected_ch0 = np.array([
        [1+2+4+5, 2+3+5+6],
        [4+5+7+8, 5+6+8+9],
    ])
    assert np.allclose(result.data[0, 0], expected_ch0)

    # Second channel: double the first
    assert np.allclose(result.data[0, 1], 2 * expected_ch0)

def test_conv2d_single_stride():
    # Single batch, single channel (1, 1, 5, 5)
    x_np = np.arange(1 * 1 * 5 * 5, dtype=float).reshape(1, 1, 5, 5)

    weight_np = np.ones((1, 1, 3, 3))
    bias_np = np.array([0.0])

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    # Stride 1
    result_s1 = conv2d_single(x, weight, bias, stride=1, padding=0)
    # Output size: (5 - 3)//1 + 1 = 3
    assert result_s1.data.shape == (1, 1, 3, 3)

    # Stride 2
    result_s2 = conv2d_single(x, weight, bias, stride=2, padding=0)
    # Output size: (5 - 3)//2 + 1 = 2
    assert result_s2.data.shape == (1, 1, 2, 2)

def test_conv2d_single_padding():
    # Single batch, single channel (1, 1, 3, 3)
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])

    weight_np = np.ones((1, 1, 3, 3))
    bias_np = np.array([0.0])

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=1)

    # With padding=1, padded input is (1, 1, 5, 5)
    # Output size: (5 - 3)//1 + 1 = 3
    assert result.data.shape == (1, 1, 3, 3)

def test_conv2d_single_backward():
    # Single batch, single channel (1, 1, 3, 3)
    x_np = np.ones((1, 1, 3, 3))
    weight_np = np.ones((1, 1, 2, 2))
    bias_np = np.array([0.0])

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=0)
    loss = result.sum()

    loss.backward()

    # All gradients should exist
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None

    # Gradients should have same shape as original tensors
    assert x.grad.shape == x_np.shape
    assert weight.grad.shape == weight_np.shape
    assert bias.grad.shape == bias_np.shape

def test_conv2d_single_batched_basic():
    # Two batches, single channel (2, 1, 3, 3)
    x_np = np.array([
        # Batch 0
        [[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]],
        # Batch 1
        [[
            [10., 11., 12.],
            [13., 14., 15.],
            [16., 17., 18.],
        ]]
    ])

    weight_np = np.ones((1, 1, 2, 2))
    bias_np = np.array([0.0])

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=0)

    # Expected shape: (2, 1, 2, 2)
    assert result.data.shape == (2, 1, 2, 2)

    # Batch 0: sum of each 2x2 patch
    expected_batch0 = np.array([[
        [1+2+4+5, 2+3+5+6],
        [4+5+7+8, 5+6+8+9],
    ]])
    assert np.allclose(result.data[0], expected_batch0)

    # Batch 1: sum of each 2x2 patch
    expected_batch1 = np.array([[
        [10+11+13+14, 11+12+14+15],
        [13+14+16+17, 14+15+17+18],
    ]])
    assert np.allclose(result.data[1], expected_batch1)

def test_conv2d_single_batched_multiple_channels():
    # Three batches, two input channels (3, 2, 3, 3)
    x_np = np.random.randn(3, 2, 3, 3)

    # Two output channels, two input channels
    weight_np = np.random.randn(2, 2, 2, 2)
    bias_np = np.random.randn(2)

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=0)

    # Expected shape: (3, 2, 2, 2) - 3 batches, 2 output channels, 2x2 output
    assert result.data.shape == (3, 2, 2, 2)

def test_conv2d_single_batched_backward():
    # Two batches, two input channels (2, 2, 4, 4)
    x_np = np.random.randn(2, 2, 4, 4)
    weight_np = np.random.randn(3, 2, 2, 2)  # 3 output channels
    bias_np = np.random.randn(3)

    x = Value(x_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result = conv2d_single(x, weight, bias, stride=1, padding=0)
    loss = result.sum()

    loss.backward()

    # All gradients should exist
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None

    # Gradients should have same shape as original tensors
    assert x.grad.shape == x_np.shape
    assert weight.grad.shape == weight_np.shape
    assert bias.grad.shape == bias_np.shape

def test_conv2d_single_batched_independence():
    # Test that batches are processed independently
    # Single batch
    x_single_np = np.random.randn(1, 2, 5, 5)

    # Two batches, where second batch is same as first
    x_batched_np = np.concatenate([x_single_np, x_single_np], axis=0)

    weight_np = np.random.randn(3, 2, 3, 3)
    bias_np = np.random.randn(3)

    x_single = Value(x_single_np)
    x_batched = Value(x_batched_np)
    weight = Value(weight_np)
    bias = Value(bias_np)

    result_single = conv2d_single(x_single, weight, bias, stride=1, padding=1)
    result_batched = conv2d_single(x_batched, weight, bias, stride=1, padding=1)

    # First batch should match single result
    assert np.allclose(result_batched.data[0], result_single.data[0])
    # Second batch should also match (since input was the same)
    assert np.allclose(result_batched.data[1], result_single.data[0])

# =============================================================================
# Tests for max() primitive
# =============================================================================

def test_max_no_axis():
    """Test max reduction over all elements."""
    x_np = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 6.0]])
    x = Value(x_np)

    result = x.max()

    expected = np.max(x_np)
    assert result.data.shape == ()  # Scalar
    assert np.allclose(result.data, expected)
    assert result.data == 6.0

def test_max_axis_0():
    """Test max reduction along axis 0."""
    x_np = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 6.0]])
    x = Value(x_np)

    result = x.max(axis=0)

    expected = np.array([4.0, 5.0, 6.0])
    assert result.data.shape == (3,)
    assert np.allclose(result.data, expected)

def test_max_axis_1():
    """Test max reduction along axis 1."""
    x_np = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 6.0]])
    x = Value(x_np)

    result = x.max(axis=1)

    expected = np.array([5.0, 6.0])
    assert result.data.shape == (2,)
    assert np.allclose(result.data, expected)

def test_max_keepdims_true():
    """Test max with keepdims=True."""
    x_np = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 6.0]])
    x = Value(x_np)

    result = x.max(axis=1, keepdims=True)

    expected = np.array([[5.0], [6.0]])
    assert result.data.shape == (2, 1)
    assert np.allclose(result.data, expected)

def test_max_backward_single_max():
    """Test backward pass when there's a unique maximum."""
    x_np = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 6.0]])
    x = Value(x_np)

    result = x.max()
    result.backward()

    # Gradient should only flow to the max element (6.0 at position [1, 2])
    expected_grad = np.array([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0]])
    assert x.grad.shape == x_np.shape
    assert np.allclose(x.grad, expected_grad)

def test_max_backward_axis():
    """Test backward pass with axis reduction."""
    x_np = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 6.0]])
    x = Value(x_np)

    result = x.max(axis=1)  # Max along rows
    loss = result.sum()  # Sum the maxes
    loss.backward()

    # Gradient should flow to max elements in each row
    # Row 0: max is 5.0 at index 1
    # Row 1: max is 6.0 at index 2
    expected_grad = np.array([[0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    assert x.grad.shape == x_np.shape
    assert np.allclose(x.grad, expected_grad)

def test_max_backward_keepdims():
    """Test backward pass with keepdims=True."""
    x_np = np.array([[1.0, 5.0, 3.0],
                     [4.0, 2.0, 6.0]])
    x = Value(x_np)

    result = x.max(axis=1, keepdims=True)
    loss = result.sum()
    loss.backward()

    # Should have same gradient pattern as without keepdims
    expected_grad = np.array([[0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]])
    assert x.grad.shape == x_np.shape
    assert np.allclose(x.grad, expected_grad)

def test_max_3d_tensor():
    """Test max on a 3D tensor."""
    x_np = np.random.randn(2, 3, 4)
    x = Value(x_np)

    result = x.max(axis=2)

    expected = np.max(x_np, axis=2)
    assert result.data.shape == (2, 3)
    assert np.allclose(result.data, expected)

def test_max_backward_tied_values():
    """Test backward when multiple elements tie for maximum."""
    x_np = np.array([[5.0, 5.0, 3.0],
                     [4.0, 2.0, 4.0]])
    x = Value(x_np)

    result = x.max(axis=1)
    loss = result.sum()
    loss.backward()

    # When there's a tie, gradient flows to all tied maximums
    # Row 0: two 5.0s at indices 0 and 1
    # Row 1: two 4.0s at indices 0 and 2
    expected_grad = np.array([[1.0, 1.0, 0.0],
                              [1.0, 0.0, 1.0]])
    assert x.grad.shape == x_np.shape
    assert np.allclose(x.grad, expected_grad)

# =============================================================================
# Tests for im2patches()
# =============================================================================

def test_im2patches_basic():
    """Test basic patch extraction."""
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # (1, 1, 3, 3)

    x = Value(x_np)
    patches = im2patches(x, kernel_size=2, stride=1)

    # Should extract 4 patches of size 2x2
    # patches shape: (B, C, Sy, Sx, k, k) = (1, 1, 2, 2, 2, 2)
    assert patches.data.shape == (1, 1, 2, 2, 2, 2)

    # Check first patch (top-left)
    expected_patch_0_0 = np.array([[1., 2.], [4., 5.]])
    assert np.allclose(patches.data[0, 0, 0, 0], expected_patch_0_0)

    # Check last patch (bottom-right)
    expected_patch_1_1 = np.array([[5., 6.], [8., 9.]])
    assert np.allclose(patches.data[0, 0, 1, 1], expected_patch_1_1)

def test_im2patches_stride():
    """Test patch extraction with different stride."""
    x_np = np.arange(1 * 1 * 5 * 5, dtype=float).reshape(1, 1, 5, 5)
    x = Value(x_np)

    # Stride 1: (5 - 3)//1 + 1 = 3 patches in each dimension
    patches_s1 = im2patches(x, kernel_size=3, stride=1)
    assert patches_s1.data.shape == (1, 1, 3, 3, 3, 3)

    # Stride 2: (5 - 3)//2 + 1 = 2 patches in each dimension
    patches_s2 = im2patches(x, kernel_size=3, stride=2)
    assert patches_s2.data.shape == (1, 1, 2, 2, 3, 3)

def test_im2patches_padding():
    """Test patch extraction with padding."""
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # (1, 1, 3, 3)

    x = Value(x_np)
    patches = im2patches(x, kernel_size=3, stride=1, padding=1)

    # With padding=1, input becomes (1, 1, 5, 5)
    # Number of patches: (5 - 3)//1 + 1 = 3 in each dimension
    assert patches.data.shape == (1, 1, 3, 3, 3, 3)

    # Top-left patch should contain zeros from padding
    # It should be:
    # [[0, 0, 0],
    #  [0, 1, 2],
    #  [0, 4, 5]]
    expected_top_left = np.array([[0., 0., 0.],
                                   [0., 1., 2.],
                                   [0., 4., 5.]])
    assert np.allclose(patches.data[0, 0, 0, 0], expected_top_left)

def test_im2patches_multi_channel():
    """Test patch extraction with multiple channels."""
    x_np = np.random.randn(1, 3, 4, 4)  # 3 channels
    x = Value(x_np)

    patches = im2patches(x, kernel_size=2, stride=1)

    # patches shape: (1, 3, 3, 3, 2, 2)
    # 3 channels, 3x3 patches, each 2x2
    assert patches.data.shape == (1, 3, 3, 3, 2, 2)

def test_im2patches_batched():
    """Test patch extraction with multiple batches."""
    x_np = np.random.randn(4, 2, 5, 5)  # 4 batches, 2 channels
    x = Value(x_np)

    patches = im2patches(x, kernel_size=3, stride=2)

    # patches shape: (4, 2, 2, 2, 3, 3)
    # 4 batches, 2 channels, 2x2 patches (stride=2), each 3x3
    assert patches.data.shape == (4, 2, 2, 2, 3, 3)

def test_im2patches_backward():
    """Test backward pass for im2patches."""
    x_np = np.arange(1 * 1 * 4 * 4, dtype=float).reshape(1, 1, 4, 4)
    x = Value(x_np)

    patches = im2patches(x, kernel_size=2, stride=1)
    loss = patches.sum()
    loss.backward()

    # Each pixel contributes to multiple patches
    # Corner pixels: 1 patch
    # Edge pixels (non-corner): 2 patches
    # Interior pixels: 4 patches
    assert x.grad is not None
    assert x.grad.shape == x_np.shape

    # For 4x4 image with 2x2 patches and stride 1:
    # participation count matrix:
    expected_counts = np.array([[
        [1., 2., 2., 1.],
        [2., 4., 4., 2.],
        [2., 4., 4., 2.],
        [1., 2., 2., 1.],
    ]])
    assert np.allclose(x.grad, expected_counts)

def test_im2patches_backward_with_padding():
    """Test backward pass with padding."""
    x_np = np.ones((1, 1, 3, 3))
    x = Value(x_np)

    patches = im2patches(x, kernel_size=3, stride=1, padding=1)
    loss = patches.sum()
    loss.backward()

    # With padding, participation counts are more uniform
    assert x.grad is not None
    assert x.grad.shape == x_np.shape

    # With padding=1 and kernel=3, stride=1 on a 3x3 input:
    # - Padded input is 5x5, we get 3x3 patches
    # - Corner pixels: participate in 4 patches
    # - Edge pixels (non-corner): participate in 6 patches
    # - Center pixel: participates in 9 patches
    expected_counts = np.array([[
        [4., 6., 4.],
        [6., 9., 6.],
        [4., 6., 4.],
    ]])
    assert np.allclose(x.grad, expected_counts)

# =============================================================================
# Tests for max_pool2d()
# =============================================================================

def test_max_pool2d_basic():
    """Test basic max pooling."""
    x_np = np.array([[
        [
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]
    ]])  # (1, 1, 4, 4)

    x = Value(x_np)
    result = max_pool2d(x, pool_size=2, stride=2)

    # With pool_size=2, stride=2, we get 2x2 output
    # Note: Looking at the implementation, H_out = (H - pool_h) // 2
    # This seems like a bug in the implementation, but let's test what it does
    # Actually checking line 187-188 in functional.py:
    # H_out = (H - pool_h) // 2
    # This should be stride, not 2
    # But let's test what the current implementation does

    # Each 2x2 region's max:
    # Top-left: max(1,2,5,6) = 6
    # Top-right: max(3,4,7,8) = 8
    # Bottom-left: max(9,10,13,14) = 14
    # Bottom-right: max(11,12,15,16) = 16
    expected = np.array([[
        [[6., 8.],
         [14., 16.]]
    ]])

    # Check shape first
    assert result.data.shape == (1, 1, 2, 2)
    assert np.allclose(result.data, expected)

def test_max_pool2d_single_channel():
    """Test max pooling on simple pattern."""
    x_np = np.array([[
        [
            [1., 5., 3., 7.],
            [4., 2., 6., 8.],
            [9., 13., 11., 15.],
            [12., 10., 14., 16.],
        ]
    ]])  # (1, 1, 4, 4)

    x = Value(x_np)
    result = max_pool2d(x, pool_size=2, stride=2)

    # Each 2x2 region's max:
    # Top-left: max(1,5,4,2) = 5
    # Top-right: max(3,7,6,8) = 8
    # Bottom-left: max(9,13,12,10) = 13
    # Bottom-right: max(11,15,14,16) = 16
    expected = np.array([[
        [[5., 8.],
         [13., 16.]]
    ]])

    assert result.data.shape == (1, 1, 2, 2)
    assert np.allclose(result.data, expected)

def test_max_pool2d_multi_channel():
    """Test max pooling with multiple channels."""
    x_np = np.random.randn(1, 3, 4, 4)  # 3 channels
    x = Value(x_np)

    result = max_pool2d(x, pool_size=2, stride=2)

    # Should have same number of channels
    assert result.data.shape == (1, 3, 2, 2)

    # Verify max pooling works per channel
    for c in range(3):
        # Top-left pool
        patch_tl = x_np[0, c, 0:2, 0:2]
        assert np.allclose(result.data[0, c, 0, 0], np.max(patch_tl))

def test_max_pool2d_batched():
    """Test max pooling with multiple batches."""
    x_np = np.random.randn(3, 2, 4, 4)  # 3 batches, 2 channels
    x = Value(x_np)

    result = max_pool2d(x, pool_size=2, stride=2)

    # Should preserve batch and channel dimensions
    assert result.data.shape == (3, 2, 2, 2)

def test_max_pool2d_backward():
    """Test backward pass for max pooling."""
    x_np = np.array([[
        [
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]
    ]])  # (1, 1, 4, 4)

    x = Value(x_np)
    result = max_pool2d(x, pool_size=2, stride=2)
    loss = result.sum()
    loss.backward()

    # Gradient should only flow to the max elements in each pool
    # Top-left pool: 6 at (0,1,1)
    # Top-right pool: 8 at (0,1,3)
    # Bottom-left pool: 14 at (0,3,1)
    # Bottom-right pool: 16 at (0,3,3)
    expected_grad = np.array([[
        [
            [0., 0., 0., 0.],
            [0., 1., 0., 1.],
            [0., 0., 0., 0.],
            [0., 1., 0., 1.],
        ]
    ]])

    assert x.grad is not None
    assert x.grad.shape == x_np.shape
    assert np.allclose(x.grad, expected_grad)

def test_max_pool2d_backward_multi_channel():
    """Test backward pass with multiple channels."""
    x_np = np.random.randn(2, 3, 4, 4)  # 2 batches, 3 channels
    x = Value(x_np)

    result = max_pool2d(x, pool_size=2, stride=2)
    loss = result.sum()
    loss.backward()

    # Gradients should exist and have same shape
    assert x.grad is not None
    assert x.grad.shape == x_np.shape

    # Gradient should be sparse (only at max locations)
    # Sum of gradients should equal number of output elements
    assert np.sum(x.grad > 0) == result.data.size

def test_max_pool2d_stride_1():
    """Test max pooling with stride=1 (overlapping pools)."""
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # (1, 1, 3, 3)

    x = Value(x_np)
    result = max_pool2d(x, pool_size=2, stride=1)

    # With stride=1, pools overlap
    # Top-left: max(1,2,4,5) = 5
    # Top-right: max(2,3,5,6) = 6
    # Bottom-left: max(4,5,7,8) = 8
    # Bottom-right: max(5,6,8,9) = 9
    expected = np.array([[
        [[5., 6.],
         [8., 9.]]
    ]])

    # Output size should be (3-2)//1 + 1 = 2 in each dimension
    assert result.data.shape == (1, 1, 2, 2)
    assert np.allclose(result.data, expected)

def test_max_pool2d_backward_overlapping():
    """Test backward with overlapping pools (stride=1)."""
    x_np = np.array([[
        [
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ]
    ]])  # (1, 1, 3, 3)

    x = Value(x_np)
    result = max_pool2d(x, pool_size=2, stride=1)
    loss = result.sum()
    loss.backward()

    # With overlapping pools, some elements might receive gradient multiple times
    # 5 is max in top-left pool
    # 6 is max in top-right pool
    # 8 is max in bottom-left pool
    # 9 is max in bottom-right pool
    expected_grad = np.array([[
        [
            [0., 0., 0.],
            [0., 1., 1.],
            [0., 1., 1.],
        ]
    ]])

    assert x.grad is not None
    assert x.grad.shape == x_np.shape
    assert np.allclose(x.grad, expected_grad)


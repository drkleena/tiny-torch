"""
Comprehensive tests for the Tensor class and all its primitives.
Tests both forward computation and gradient backpropagation.
"""

import numpy as np
import pytest
from autograd.engine import Tensor


# ============================================================================
# Helper Functions
# ============================================================================

def numerical_gradient(func, x, eps=1e-5):
    """
    Compute numerical gradient using finite differences.
    func: scalar function taking numpy array
    x: numpy array input
    returns: numerical gradient with same shape as x
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        x[idx] = old_value + eps
        fxh_pos = func(x)

        x[idx] = old_value - eps
        fxh_neg = func(x)

        x[idx] = old_value

        grad[idx] = (fxh_pos - fxh_neg) / (2 * eps)
        it.iternext()

    return grad


def check_gradient(val: Tensor, expected_grad: np.ndarray, rtol=1e-5, atol=1e-5):
    """Helper to check if gradient matches expected."""
    assert np.allclose(val.grad, expected_grad, rtol=rtol, atol=atol), \
        f"Gradient mismatch:\nExpected:\n{expected_grad}\nGot:\n{val.grad}"


# ============================================================================
# Basic Arithmetic Operations
# ============================================================================

def test_add_forward():
    """Test addition forward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a + b

    expected = np.array([[6.0, 8.0], [10.0, 12.0]])
    assert np.allclose(c.data, expected)


def test_add_backward():
    """Test addition backward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a + b
    loss = c.sum()

    loss.backward()

    # Gradient of sum(a + b) w.r.t. a is all ones
    check_gradient(a, np.ones_like(a.data))
    check_gradient(b, np.ones_like(b.data))


def test_add_scalar():
    """Test adding scalar to Tensor."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a + 5.0

    expected = np.array([[6.0, 7.0], [8.0, 9.0]])
    assert np.allclose(b.data, expected)


def test_radd():
    """Test reverse addition (scalar + Tensor)."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = 5.0 + a

    expected = np.array([[6.0, 7.0], [8.0, 9.0]])
    assert np.allclose(b.data, expected)


def test_add_broadcasting():
    """Test addition with broadcasting."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = Tensor([[10.0, 20.0]])  # (1, 2)
    c = a + b

    expected = np.array([[11.0, 22.0], [13.0, 24.0]])
    assert np.allclose(c.data, expected)

    # Test gradients with broadcasting
    loss = c.sum()
    loss.backward()

    # a gets gradient of ones
    check_gradient(a, np.ones_like(a.data))
    # b gets gradient summed over broadcast dimension
    check_gradient(b, np.array([[2.0, 2.0]]))


def test_mul_forward():
    """Test multiplication forward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 3.0], [4.0, 5.0]])
    c = a * b

    expected = np.array([[2.0, 6.0], [12.0, 20.0]])
    assert np.allclose(c.data, expected)


def test_mul_backward():
    """Test multiplication backward pass."""
    a = Tensor([[2.0, 3.0]])
    b = Tensor([[4.0, 5.0]])
    c = a * b
    loss = c.sum()

    loss.backward()

    # d(a*b)/da = b
    check_gradient(a, b.data)
    # d(a*b)/db = a
    check_gradient(b, a.data)


def test_rmul():
    """Test reverse multiplication."""
    a = Tensor([[2.0, 3.0]])
    b = 5.0 * a

    expected = np.array([[10.0, 15.0]])
    assert np.allclose(b.data, expected)


def test_neg():
    """Test negation."""
    a = Tensor([[1.0, -2.0], [3.0, -4.0]])
    b = -a

    expected = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    assert np.allclose(b.data, expected)

    # Test gradient
    loss = b.sum()
    loss.backward()
    check_gradient(a, np.full_like(a.data, -1.0))


def test_sub():
    """Test subtraction."""
    a = Tensor([[5.0, 6.0]])
    b = Tensor([[2.0, 3.0]])
    c = a - b

    expected = np.array([[3.0, 3.0]])
    assert np.allclose(c.data, expected)

    # Test gradient
    loss = c.sum()
    loss.backward()
    check_gradient(a, np.ones_like(a.data))
    check_gradient(b, -np.ones_like(b.data))


def test_rsub():
    """Test reverse subtraction."""
    a = Tensor([[2.0, 3.0]])
    b = 10.0 - a

    expected = np.array([[8.0, 7.0]])
    assert np.allclose(b.data, expected)


def test_pow():
    """Test power operation."""
    a = Tensor([[2.0, 3.0], [4.0, 5.0]])
    b = a ** 2

    expected = np.array([[4.0, 9.0], [16.0, 25.0]])
    assert np.allclose(b.data, expected)

    # Test gradient: d(x^2)/dx = 2x
    loss = b.sum()
    loss.backward()
    check_gradient(a, 2 * a.data)


def test_pow_fractional():
    """Test fractional power."""
    a = Tensor([[4.0, 9.0, 16.0]])
    b = a ** 0.5

    expected = np.array([[2.0, 3.0, 4.0]])
    assert np.allclose(b.data, expected)


def test_truediv():
    """Test division."""
    a = Tensor([[6.0, 8.0]])
    b = Tensor([[2.0, 4.0]])
    c = a / b

    expected = np.array([[3.0, 2.0]])
    assert np.allclose(c.data, expected)

    # Test gradient
    loss = c.sum()
    loss.backward()
    # d(a/b)/da = 1/b
    check_gradient(a, 1.0 / b.data)
    # d(a/b)/db = -a/b^2
    check_gradient(b, -a.data / (b.data ** 2))


def test_rtruediv():
    """Test reverse division."""
    a = Tensor([[2.0, 4.0]])
    b = 12.0 / a

    expected = np.array([[6.0, 3.0]])
    assert np.allclose(b.data, expected)


# ============================================================================
# Matrix Operations
# ============================================================================

def test_matmul_forward():
    """Test matrix multiplication forward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)
    c = a @ b

    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(c.data, expected)


def test_matmul_backward():
    """Test matrix multiplication backward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    loss = c.sum()

    loss.backward()

    # dc/da = out.grad @ b^T
    expected_grad_a = np.ones((2, 2)) @ b.data.T
    check_gradient(a, expected_grad_a)

    # dc/db = a^T @ out.grad
    expected_grad_b = a.data.T @ np.ones((2, 2))
    check_gradient(b, expected_grad_b)


def test_matmul_batch():
    """Test batched matrix multiplication."""
    # Batch of 3, each (2, 3) @ (3, 4)
    a = Tensor(np.random.randn(2, 3))
    b = Tensor(np.random.randn(3, 4))
    c = a @ b

    assert c.data.shape == (2, 4)
    expected = a.data @ b.data
    assert np.allclose(c.data, expected)


# ============================================================================
# Activation Functions
# ============================================================================

def test_relu_forward():
    """Test ReLU forward pass."""
    a = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    b = a.relu()

    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
    assert np.allclose(b.data, expected)


def test_relu_backward():
    """Test ReLU backward pass."""
    a = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    b = a.relu()
    loss = b.sum()

    loss.backward()

    # Gradient is 1 where input > 0, else 0
    expected_grad = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    check_gradient(a, expected_grad)


def test_tanh_forward():
    """Test tanh forward pass."""
    a = Tensor([[0.0, 1.0, -1.0]])
    b = a.tanh()

    expected = np.tanh(a.data)
    assert np.allclose(b.data, expected)


def test_tanh_backward():
    """Test tanh backward pass."""
    a_data = np.array([[0.5, 1.0]])
    a = Tensor(a_data)
    b = a.tanh()
    loss = b.sum()

    loss.backward()

    # d(tanh(x))/dx = 1 - tanh(x)^2
    tanh_val = np.tanh(a_data)
    expected_grad = 1 - tanh_val ** 2
    check_gradient(a, expected_grad)


def test_exp_forward():
    """Test exp forward pass."""
    a = Tensor([[0.0, 1.0, 2.0]])
    b = a.exp()

    expected = np.exp(a.data)
    assert np.allclose(b.data, expected)


def test_exp_backward():
    """Test exp backward pass."""
    a_data = np.array([[1.0, 2.0]])
    a = Tensor(a_data)
    b = a.exp()
    loss = b.sum()

    loss.backward()

    # d(exp(x))/dx = exp(x)
    expected_grad = np.exp(a_data)
    check_gradient(a, expected_grad)


def test_log_forward():
    """Test log forward pass."""
    a = Tensor([[1.0, 2.0, np.e]])
    b = a.log()

    expected = np.log(a.data)
    assert np.allclose(b.data, expected)


def test_log_backward():
    """Test log backward pass."""
    a_data = np.array([[1.0, 2.0, 4.0]])
    a = Tensor(a_data)
    b = a.log()
    loss = b.sum()

    loss.backward()

    # d(log(x))/dx = 1/x
    expected_grad = 1.0 / a_data
    check_gradient(a, expected_grad)


# ============================================================================
# Reduction Operations
# ============================================================================

def test_sum_all():
    """Test sum over all elements."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.sum()

    assert np.allclose(b.data, 10.0)

    # Gradient should be all ones
    b.backward()
    check_gradient(a, np.ones_like(a.data))


def test_sum_axis0():
    """Test sum over axis 0."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.sum(axis=0)

    expected = np.array([4.0, 6.0])
    assert np.allclose(b.data, expected)

    # Test gradient
    loss = b.sum()
    loss.backward()
    check_gradient(a, np.ones_like(a.data))


def test_sum_axis1_keepdims():
    """Test sum with keepdims."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.sum(axis=1, keepdims=True)

    expected = np.array([[3.0], [7.0]])
    assert np.allclose(b.data, expected)
    assert b.data.shape == (2, 1)


def test_mean_all():
    """Test mean over all elements."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.mean()

    assert np.allclose(b.data, 2.5)

    # Gradient should be 1/n for each element
    loss = b * 1  # Keep as is
    loss.backward()
    check_gradient(a, np.full_like(a.data, 0.25))


def test_mean_axis():
    """Test mean over specific axis."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.mean(axis=0)

    expected = np.array([2.0, 3.0])
    assert np.allclose(b.data, expected)


def test_max_all():
    """Test max over all elements."""
    a = Tensor([[1.0, 5.0], [3.0, 2.0]])
    b = a.max()

    assert np.allclose(b.data, 5.0)

    # Gradient should flow only to max element
    b.backward()
    expected_grad = np.array([[0.0, 1.0], [0.0, 0.0]])
    check_gradient(a, expected_grad)


def test_max_axis():
    """Test max over specific axis."""
    a = Tensor([[1.0, 5.0], [3.0, 2.0]])
    b = a.max(axis=1)

    expected = np.array([5.0, 3.0])
    assert np.allclose(b.data, expected)

    # Test gradient
    loss = b.sum()
    loss.backward()
    expected_grad = np.array([[0.0, 1.0], [1.0, 0.0]])
    check_gradient(a, expected_grad)


def test_max_tied_values():
    """Test max with tied values (gradient should be split)."""
    a = Tensor([[2.0, 2.0, 1.0]])
    b = a.max(axis=1)

    assert np.allclose(b.data, 2.0)

    # Both max values should get gradient
    b.backward()
    expected_grad = np.array([[1.0, 1.0, 0.0]])
    check_gradient(a, expected_grad)


# ============================================================================
# Shape Operations
# ============================================================================

def test_reshape_forward():
    """Test reshape forward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.reshape((4, 1))

    expected = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert np.allclose(b.data, expected)


def test_reshape_backward():
    """Test reshape backward pass."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.reshape((4, 1))
    loss = b.sum()

    loss.backward()
    check_gradient(a, np.ones_like(a.data))


def test_transpose_2d():
    """Test 2D transpose."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.transpose()

    expected = np.array([[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(b.data, expected)

    # Test gradient
    loss = b.sum()
    loss.backward()
    check_gradient(a, np.ones_like(a.data))


def test_transpose_property():
    """Test .T property."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.T

    expected = a.transpose().data
    assert np.allclose(b.data, expected)


def test_transpose_3d():
    """Test 3D transpose with axes."""
    a = Tensor(np.arange(24).reshape(2, 3, 4))
    b = a.transpose((2, 0, 1))  # (4, 2, 3)

    expected = np.transpose(a.data, (2, 0, 1))
    assert np.allclose(b.data, expected)
    assert b.data.shape == (4, 2, 3)


def test_pad_forward():
    """Test padding forward pass."""
    # Create a (C, H, W) = (1, 2, 2) tensor
    a = Tensor([
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    ])  # (1, 2, 2)

    b = a.pad(1, 1)  # Pad (1, 2, 2) -> (1, 4, 4)

    assert b.data.shape == (1, 4, 4)
    # Check corners are zero
    assert b.data[0, 0, 0] == 0.0
    assert b.data[0, 0, -1] == 0.0
    # Check center values preserved
    assert b.data[0, 1, 1] == 1.0
    assert b.data[0, 2, 2] == 4.0


def test_pad_backward():
    """Test padding backward pass."""
    # Create a (C, H, W) = (1, 2, 2) tensor
    a_data = np.array([
        [
            [1.0, 2.0],
            [3.0, 4.0]
        ]
    ])  # (1, 2, 2)
    a = Tensor(a_data)

    b = a.pad(1, 1)
    loss = b.sum()

    loss.backward()
    # Gradient should flow back to original positions
    check_gradient(a, np.ones_like(a.data))


# ============================================================================
# Indexing and Stacking
# ============================================================================

def test_getitem_basic():
    """Test basic indexing."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a[0]

    expected = np.array([1.0, 2.0])
    assert np.allclose(b.data, expected)


def test_getitem_slice():
    """Test slice indexing."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a[:, 1:]

    expected = np.array([[2.0, 3.0], [5.0, 6.0]])
    assert np.allclose(b.data, expected)


def test_getitem_backward():
    """Test indexing backward pass."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a[:, 1:]
    loss = b.sum()

    loss.backward()

    # Only indexed elements should receive gradient
    expected_grad = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    check_gradient(a, expected_grad)


def test_stack_basic():
    """Test stacking Values."""
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0, 4.0]])
    c = Tensor.stack([a, b], axis=0)

    # Stack adds a new dimension, so (1,2) stacked along axis 0 becomes (2,1,2)
    expected = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    assert np.allclose(c.data, expected)
    assert c.data.shape == (2, 1, 2)


def test_stack_backward():
    """Test stack backward pass."""
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0, 4.0]])
    c = Tensor.stack([a, b], axis=0)
    loss = c.sum()

    loss.backward()

    check_gradient(a, np.ones_like(a.data))
    check_gradient(b, np.ones_like(b.data))


def test_concat_basic():
    """Test concatenation."""
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0, 4.0]])
    c = Tensor.concat([a, b], axis=0)

    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(c.data, expected)


def test_concat_axis1():
    """Test concatenation along axis 1."""
    a = Tensor([[1.0], [2.0]])
    b = Tensor([[3.0], [4.0]])
    c = Tensor.concat([a, b], axis=1)

    expected = np.array([[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(c.data, expected)


def test_concat_backward():
    """Test concat backward pass."""
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0, 4.0]])
    c = Tensor.concat([a, b], axis=0)
    loss = c.sum()

    loss.backward()

    check_gradient(a, np.ones_like(a.data))
    check_gradient(b, np.ones_like(b.data))


# ============================================================================
# Comparison and Clipping
# ============================================================================

def test_gt():
    """Test greater-than comparison."""
    a = Tensor([[1.0, 2.0, 3.0]])
    b = a > 2.0

    expected = np.array([[0.0, 0.0, 1.0]])
    assert np.allclose(b.data, expected)


def test_lt():
    """Test less-than comparison."""
    a = Tensor([[1.0, 2.0, 3.0]])
    b = a < 2.0

    expected = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(b.data, expected)


def test_clip_forward():
    """Test clip forward pass."""
    a = Tensor([[-2.0, 0.0, 5.0, 10.0]])
    b = a.clip(0.0, 5.0)

    expected = np.array([[0.0, 0.0, 5.0, 5.0]])
    assert np.allclose(b.data, expected)


def test_clip_backward():
    """Test clip backward pass (gradients flow only in valid range)."""
    a = Tensor([[-2.0, 1.0, 3.0, 10.0]])
    b = a.clip(0.0, 5.0)
    loss = b.sum()

    loss.backward()

    # Gradient flows only where not clipped
    expected_grad = np.array([[0.0, 1.0, 1.0, 0.0]])
    check_gradient(a, expected_grad)


# ============================================================================
# Complex Gradient Tests
# ============================================================================

def test_complex_computation_graph():
    """Test a complex computation graph with multiple operations."""
    a = Tensor([[2.0, 3.0]])
    b = Tensor([[4.0, 5.0]])

    # c = (a * b) + (a ** 2)
    c = (a * b) + (a ** 2)
    loss = c.sum()

    loss.backward()

    # dc/da = b + 2*a
    expected_grad_a = b.data + 2 * a.data
    check_gradient(a, expected_grad_a)

    # dc/db = a
    expected_grad_b = a.data
    check_gradient(b, expected_grad_b)


def test_chain_of_operations():
    """Test gradient flow through a chain of operations."""
    a = Tensor([[2.0]])
    b = a * 2
    c = b + 3
    d = c ** 2
    e = d.sum()

    # e = ((a * 2) + 3)^2
    # de/da = 2 * ((a * 2) + 3) * 2 = 4 * (2a + 3)
    e.backward()

    expected = 4 * (2 * a.data + 3)
    check_gradient(a, expected)


def test_multiple_paths_gradient_accumulation():
    """Test that gradients accumulate when variable is used multiple times."""
    a = Tensor([[3.0]])
    b = a + a  # a is used twice
    loss = b.sum()

    loss.backward()

    # Both paths contribute gradient of 1, total = 2
    check_gradient(a, np.array([[2.0]]))


def test_broadcasting_complex():
    """Test complex broadcasting scenario."""
    a = Tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    b = Tensor([[1.0], [2.0]])  # (2, 1)

    c = a * b  # Broadcasts to (2, 3)
    loss = c.sum()

    loss.backward()

    # a gradient: sum over broadcast dim (axis 0)
    expected_grad_a = np.sum(b.data, axis=0, keepdims=True)
    check_gradient(a, expected_grad_a)

    # b gradient: sum over broadcast dim (axis 1)
    expected_grad_b = np.sum(a.data, axis=1, keepdims=True)
    check_gradient(b, expected_grad_b)


# ============================================================================
# Edge Cases
# ============================================================================

def test_scalar_value():
    """Test operations on scalar Values."""
    a = Tensor(5.0)
    b = Tensor(3.0)
    c = a * b

    assert np.allclose(c.data, 15.0)

    c.backward()
    check_gradient(a, np.array(3.0))
    check_gradient(b, np.array(5.0))


def test_zero_gradient():
    """Test operations that should produce zero gradient."""
    a = Tensor([[1.0, 2.0, 3.0]])
    b = a.relu()
    c = b * 0  # Multiply by zero
    loss = c.sum()

    loss.backward()

    # Gradient should be zero since multiplied by zero
    check_gradient(a, np.zeros_like(a.data))


def test_length():
    """Test __len__ method."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert len(a) == 2


def test_repr():
    """Test __repr__ method."""
    a = Tensor([[1.0, 2.0]])
    repr_str = repr(a)
    assert "1." in repr_str and "2." in repr_str


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

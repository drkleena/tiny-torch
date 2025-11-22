import numpy as np
from autograd.engine import Value
from nn.functional import im2col  # adjust import to your layout

def test_im2col_single_channel_3x3_k2_s1():
    X_np = np.array([[
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
    ]])  # shape (1,3,3)

    X = Value(X_np)
    cols = im2col(X, kernel_size=2, stride=1)

    expected = np.array([
        [1., 2., 4., 5.],
        [2., 3., 5., 6.],
        [4., 5., 7., 8.],
        [5., 6., 8., 9.],
    ])  # shape (4,4)

    assert cols.data.shape == expected.shape
    assert np.allclose(cols.data, expected)

def test_im2col_two_channels_3x3_k2_s1():
    X_np = np.array([
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
    ])  # shape (2,3,3)

    X = Value(X_np)
    cols = im2col(X, kernel_size=2, stride=1)

    # Expected: each column is [C0_patch_flat; C1_patch_flat]
    expected = np.array([
        [ 1.,  2.,  4.,  5.],
        [ 2.,  3.,  5.,  6.],
        [ 4.,  5.,  7.,  8.],
        [ 5.,  6.,  8.,  9.],
        [11., 12., 14., 15.],
        [12., 13., 15., 16.],
        [14., 15., 17., 18.],
        [15., 16., 18., 19.],
    ])  # shape (8,4)

    assert cols.data.shape == expected.shape
    assert np.allclose(cols.data, expected)

def test_im2col_stride_shape():
    X_np = np.arange(1 * 5 * 5, dtype=float).reshape(1, 5, 5)
    X = Value(X_np)

    k = 3

    cols_s1 = im2col(X, kernel_size=k, stride=1)
    cols_s2 = im2col(X, kernel_size=k, stride=2)

    # H_out = (H - k)//stride + 1
    H_out_s1 = (5 - k)//1 + 1  # 3
    H_out_s2 = (5 - k)//2 + 1  # 2

    # rows = C * k * k = 1 * 3 * 3 = 9
    assert cols_s1.data.shape == (9, H_out_s1 * H_out_s1)  # (9, 9)
    assert cols_s2.data.shape == (9, H_out_s2 * H_out_s2)  # (9, 4)

def test_im2col_backward_overlaps():
    X_np = np.arange(1 * 3 * 3, dtype=float).reshape(1, 3, 3)
    X = Value(X_np)

    cols = im2col(X, kernel_size=2, stride=1)  # (4,4)
    loss = cols.sum()                          # scalar

    loss.backward()

    # Each pixel participation count:
    # corners: 1, edges (non-corner): 2, center: 4
    expected_counts = np.array([[
        [1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.],
    ]])

    assert X.grad.shape == expected_counts.shape
    assert np.allclose(X.grad, expected_counts)

def test_im2col_conv_equivalence():
    X_np = np.array([[
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
    ]])  # (1,3,3)

    X = Value(X_np)

    k = 2
    cols = im2col(X, kernel_size=k, stride=1)   # (4,4)

    # Single filter, all ones → sums of each 2x2 patch
    W_np = np.ones((1, 1, k, k), dtype=float)
    W = Value(W_np)

    W_mat = W.reshape((1, 1 * k * k))          # (1,4)
    Y_cols = W_mat @ cols                      # (1,4)
    Y = Y_cols.reshape((1, 2, 2))              # (1,2,2)

    expected = np.array([[
        [1+2+4+5, 2+3+5+6],
        [4+5+7+8, 5+6+8+9],
    ]])
    
    assert np.allclose(Y.data, expected)

def test_im2col_padding_forward():
    X_np = np.array([[
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
    ]])  # (1,3,3)

    X = Value(X_np)
    cols = im2col(X, kernel_size=3, stride=1, padding=1)

    # padded input is (1,5,5)
    # H_out = W_out = (5 - 3)//1 + 1 = 3
    # rows = C*k*k = 1*3*3 = 9
    assert cols.data.shape == (9, 9)

def test_im2col_padding_backward():
    X_np = np.array([[
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
    ]])  # (1,3,3)

    X = Value(X_np)

    # single padding stage: let im2col handle pad
    cols = im2col(X, kernel_size=3, stride=1, padding=1)
    loss = cols.sum()
    loss.backward()

    expected = np.array([[
        [4., 6., 4.],
        [6., 9., 6.],
        [4., 6., 4.],
    ]])  # shape (1,3,3) to match X.grad

    # or Option 2: exact equality
    assert (X.grad == expected).all()

def test():
    test_im2col_single_channel_3x3_k2_s1()
    test_im2col_two_channels_3x3_k2_s1()
    test_im2col_stride_shape()
    test_im2col_backward_overlaps()
    test_im2col_conv_equivalence()
    test_im2col_padding_forward()
    test_im2col_padding_backward()

if __name__ == "__main__":
    test()
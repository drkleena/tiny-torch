# Tiny Micro Torch

A minimal, educational implementation of an automatic differentiation engine and neural network library, inspired by PyTorch. Built from scratch with NumPy, supporting both fully connected and convolutional neural networks.

## Project Structure

```
tiny-micro-torch/
   autograd/                # Autograd engine (core differentiation)
         engine.py           # Tensor class with tensor operations and backward()
         __init__.py
   nn/                      # Neural network modules
      modules.py           # Linear, Conv2D, MaxPool2D, Flatten, Network
      functional.py        # Stateless functions (conv2d, softmax, cross_entropy, etc.)
      __init__.py
   optim/                   # Optimizers
         sgd.py             # Stochastic Gradient Descent
         adam.py            # Adam optimizer
      __init__.py
   data/                    # Data loading utilities
         mnist.py           # MNIST dataset loader
      __init__.py
   reporters/               # Training reporters
         wandb_reporter.py  # Weights & Biases integration
      __init__.py
   examples/                # Training examples
      mnist_mlp.py         # MNIST MLP classification
      mnist_conv.py        # MNIST CNN classification
      mnist_conv_keras_benchmark.py  # Keras comparison
      autoencoder.py       # Autoencoder example
   tests/                   # Unit tests
   README.md
```

## Features

### Autograd Engine (`autograd/engine.py`)
- **Tensor class**: Wraps NumPy arrays with automatic differentiation
- **Operations**: `+`, `-`, `*`, `/`, `**`, `@` (matmul)
- **Activations**: `tanh()`, `relu()`, `exp()`, `log()`, `sigmoid()`
- **Reductions**: `sum()`, `mean()`, `max()`
- **Tensor operations**: `reshape()`, `transpose()`, `pad()`, `clip()`, `stack()`
- **Indexing**: Advanced NumPy-style indexing with gradient support
- **Automatic broadcasting**: Handles shape mismatches correctly
- **Backward pass**: Topological sort + chain rule

### Neural Network API (`nn/`)
- **Linear layer**: Fully connected layer with optional activation
- **Conv2D layer**: 2D convolutional layer with customizable kernel, stride, and padding
- **MaxPool2D layer**: Max pooling layer for dimensionality reduction
- **Flatten layer**: Flattens spatial dimensions for transitioning to fully connected layers
- **Network**: Sequential container for stacking layers
- **Loss functions**: Cross-entropy, MSE, binary cross-entropy
- **Functional ops**:
  - Convolution: `conv2d()`, `im2col()`, `im2patches()`, `max_pool2d()`
  - Activations: `softmax()`, `sigmoid()`
  - Loss: `cross_entropy()`, `mse()`, `binary_cross_entropy()`

### Optimizers (`optim/`)
- **SGD**: Stochastic Gradient Descent with configurable learning rate
- **Adam**: Adaptive Moment Estimation optimizer with bias correction

### Data Utilities (`data/`)
- **MNIST loader**: Automatic download, preprocessing, and one-hot encoding

### Training Utilities (`reporters/`)
- **WandB Reporter**: Integration with Weights & Biases for experiment tracking

## Quick Start

### Training a Neural Network on MNIST

```python
from autograd import Tensor
from nn import Network, Linear, cross_entropy
from optim import SGD
from data import load_mnist

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Build model
model = Network([
    Linear(784, 128, activation="relu"),
    Linear(128, 64, activation="relu"),
    Linear(64, 10, activation=None),  # logits
])

# Create optimizer
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10):
    # Get batch (simplified)
    X_batch = Tensor(X_train[:1000])
    y_batch = Tensor(y_train[:1000])

    # Forward pass
    logits = model(X_batch)
    loss = cross_entropy(logits, y_batch)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

### Training a CNN on MNIST

```python
from autograd import Tensor
from nn import Network, Conv2D, MaxPool2D, Flatten, Linear, cross_entropy
from optim import Adam
from data import load_mnist

# Load data
X_train, y_train, X_test, y_test = load_mnist()

# Reshape for CNN: (N, 1, 28, 28)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# Build CNN model
model = Network([
    Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, activation="relu"),
    MaxPool2D(pool_size=2, stride=2),
    Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, activation="relu"),
    MaxPool2D(pool_size=2, stride=2),
    Flatten(),
    Linear(32 * 7 * 7, 128, activation="relu"),
    Linear(128, 10, activation=None),  # logits
])

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop (similar to MLP example)
# ...
```

### Running the Examples

```bash
# Train MLP
python examples/mnist_mlp.py

# Train CNN
python examples/mnist_conv.py

# Compare with Keras
python examples/mnist_conv_keras_benchmark.py
```

The MLP achieves ~95% test accuracy, while the CNN reaches higher accuracy with more robust feature extraction.

## Design Philosophy

This project follows a clean separation between:

1. **Engine layer** (`autograd/`): Pure autodiff engine
   - No knowledge of neural networks or optimizers
   - Just Tensor nodes, operations, and backward propagation

2. **Library layer** (`nn/`, `optim/`): PyTorch-like user API
   - Modules, layers, loss functions
   - Optimizers for parameter updates
   - Built on top of the engine

3. **Examples** (`examples/`): Real training scripts
   - Demonstrates how to use the library
   - Trains actual models on real datasets

## Key Implementation Details

### Automatic Broadcasting
The engine correctly handles NumPy broadcasting in operations like addition and multiplication, with proper gradient accumulation using the `_unbroadcast` helper.

### Topological Sorting
The `backward()` method builds a topological ordering of the computation graph to ensure gradients flow in the correct order.

### Memory Efficiency
Gradients accumulate additively (using `+=`) to support:
- Parameter reuse in computation graphs
- Proper gradient flow through sum/mean operations
- Correct handling of broadcast operations

### Convolutional Neural Networks
The library implements CNNs using the **im2col** (image-to-column) algorithm:
- Converts convolution operations into efficient matrix multiplications
- `im2col()` extracts sliding window patches from input images
- `im2patches()` provides batched patch extraction for pooling operations
- Supports configurable stride, padding, and kernel sizes
- Proper gradient backpropagation through all convolutional layers

The implementation is fully differentiable and compatible with the autograd engine, enabling end-to-end training of CNNs.

## Requirements

```
numpy
tensorflow  # Only for loading MNIST
tqdm        # Progress bars
wandb       # Experiment tracking (optional)
matplotlib  # Plotting (optional)
pandas      # Loss smoothing (optional)
```

## Educational Goals

This project is designed to help understand:
- How automatic differentiation works under the hood
- The relationship between computation graphs and gradients
- How PyTorch-like frameworks are structured
- The basics of training neural networks from scratch
- How convolutional neural networks are implemented using im2col
- The mechanics of backpropagation through complex tensor operations
- How different optimizers (SGD vs Adam) affect training dynamics

## Limitations

This is an educational project. For production use, stick with PyTorch, JAX, or TensorFlow. Known limitations:
- No GPU support
- No computational graph optimization
- Limited operation set compared to major frameworks
- Not memory-optimized for large models
- Conv2D implementation uses im2col (memory intensive for large feature maps)
- Limited activation functions and layer types

## License

MIT

## Acknowledgments

Inspired by:
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- PyTorch's design and API
- [Tinygrad](https://github.com/tinygrad/tinygrad)
- [hips/autograd](https://github.com/HIPS/autograd)

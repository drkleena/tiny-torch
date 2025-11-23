"""
MNIST MLP Training Example

This script demonstrates how to train a multi-layer perceptron (MLP)
on the MNIST dataset using tiny-micro-torch.

The model is a simple feedforward network with:
- Input layer: 784 features (28x28 flattened images)
- Hidden layer 1: 128 units with ReLU activation
- Hidden layer 2: 64 units with ReLU activation
- Output layer: 10 units (logits for 10 classes)
"""

import numpy as np
from tqdm import tqdm

from autograd import Tensor
from nn import Network, Linear, cross_entropy
from optim import SGD
from data import load_mnist
from reporters import TrainingReporter


def train_mnist(epochs=60, batch_size=1000, lr=0.1, reporter=None):
    """
    Train an MLP on MNIST dataset.

    Args:
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        lr: Learning rate for SGD optimizer
        reporter: Optional TrainingReporter instance for logging metrics

    Returns:
        tuple: (model, losses) - trained model and list of loss values
    """
    # Load MNIST data
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Build model
    print("\nBuilding model...")
    model = Network([
        Linear(784, 128, activation="relu"),
        Linear(128, 64, activation="relu"),
        Linear(64, 10, activation=None),   # logits layer (no activation)
    ])

    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=lr)

    # Training setup
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    losses = []

    print(f"\nTraining on {n_samples} samples for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Batches per epoch: {n_batches}")
    print(f"Learning rate: {lr}\n")

    # Training loop
    for epoch in range(epochs):
        # Shuffle training data
        permutation = np.random.permutation(n_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_losses = []

        # Mini-batch training
        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            # Get batch
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Wrap in Tensor
            X_batch = Tensor(X_batch)
            y_batch = Tensor(y_batch)

            # Forward pass
            logits = model(X_batch)
            loss = cross_entropy(logits, y_batch)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

            # Record loss
            batch_loss = float(loss.data)
            epoch_losses.append(batch_loss)
            losses.append(batch_loss)

            # Log to reporter
            if reporter:
                reporter.log_batch(
                    loss=batch_loss,
                    batch_idx=batch_idx,
                    epoch=epoch
                )

        # Print epoch summary
        avg_loss = np.mean(epoch_losses)
        accuracy = evaluate_mnist(model, X_test, y_test, reporter=reporter)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {accuracy:.4f}")
        # Log epoch metrics
        if reporter:
            reporter.log_epoch(avg_loss=avg_loss, epoch=epoch)

    print("\nTraining complete!")
    return model, losses


def evaluate_mnist(model, X_test, y_test, batch_size=1000, reporter=None, split="test"):
    """
    Evaluate model accuracy on test set.

    Args:
        model: Trained Network model
        X_test: Test images, shape (n_samples, 784)
        y_test: Test labels, shape (n_samples, 10), one-hot encoded
        batch_size: Batch size for evaluation
        reporter: Optional TrainingReporter instance for logging metrics
        split: Name of the data split being evaluated (e.g., "test", "val")

    Returns:
        float: Accuracy on test set (0-1)
    """
    n_samples = X_test.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    correct = 0
    total = 0

    print(f"\nEvaluating on {n_samples} test samples...")

    for batch_idx in tqdm(range(n_batches), desc="Evaluating"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        X_batch = Tensor(X_test[start:end])
        y_batch = y_test[start:end]

        # Forward pass
        logits = model(X_batch)

        # Get predictions (argmax of logits)
        predictions = np.argmax(logits.data, axis=1)
        targets = np.argmax(y_batch, axis=1)

        correct += np.sum(predictions == targets)
        total += len(predictions)

    accuracy = correct / total
    print(f"\n{split.capitalize()} Accuracy: {accuracy:.4f} ({correct}/{total} correct)")

    # Log evaluation metrics
    if reporter:
        reporter.log_evaluation(accuracy=accuracy, split=split)

    return accuracy


if __name__ == "__main__":
    # Training configuration
    config = {
        "epochs": 60,
        "batch_size": 1000,
        "lr": 0.2,
        "architecture": "MLP",
        "hidden_layers": [128, 64],
        "activation": "relu"
    }

    # Initialize wandb reporter (set enabled=False to disable wandb logging)
    reporter = TrainingReporter(
        project="tiny-micro-torch-mnist",
        name="mnist-mlp-baseline",
        config=config,
        tags=["mnist", "mlp", "baseline"],
        enabled=True  # Set to False to disable wandb
    )

    # Train the model
    model, losses = train_mnist(
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        reporter=reporter
    )

    # Load test data for evaluation
    _, _, X_test, y_test = load_mnist()

    # Evaluate on test set
    

    # Finish wandb logging
    reporter.finish()

    # Optional: Plot loss curve
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        # Smooth the loss curve
        df = pd.DataFrame({'loss': losses})
        df['loss_smooth'] = df['loss'].ewm(alpha=0.1).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(df['loss_smooth'], label='Smoothed Loss', linewidth=2)
        plt.plot(df['loss'], label='Raw Loss', alpha=0.3, linewidth=1)
        plt.xlabel('Batch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('mnist_training_loss.png')
        print("\nLoss plot saved as 'mnist_training_loss.png'")
    except ImportError:
        print("\nMatplotlib/pandas not available - skipping loss plot")

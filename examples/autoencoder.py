import numpy as np
from tqdm import tqdm
import wandb

from autograd import Value
from nn import Network, Linear, mse, binary_cross_entropy
from optim import SGD, Adam
from data import load_mnist
from reporters import TrainingReporter


def train_autoencoder(epochs=60, batch_size=1000, lr=0.1, reporter=None):
    """
    Train an autoencoder on MNIST dataset.

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
        Linear(784, 32, activation="relu"),      # encoder
        Linear(32, 784, activation="sigmoid"),   # decoder
    ])

    # Initialize optimizer
    # optimizer = SGD(model.parameters(), lr=lr)
    optimizer = Adam(model.parameters(), lr=lr)

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

            # Wrap in Value
            X_batch = Value(X_batch)

            # Forward pass
            reconstructed = model(X_batch)
            loss = binary_cross_entropy(reconstructed, X_batch)

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

                # Log reconstruction examples (5 random samples per batch)
                n_examples = min(5, len(X_batch.data))
                random_indices = np.random.choice(len(X_batch.data), size=n_examples, replace=False)

                # Create comparison images (original vs reconstruction)
                wandb_images = []
                for idx in random_indices:
                    original = X_batch.data[idx].reshape(28, 28)
                    reconstructed_img = reconstructed.data[idx].reshape(28, 28)

                    # Stack original and reconstruction side by side
                    comparison = np.hstack([original, reconstructed_img])
                    wandb_images.append(
                        wandb.Image(comparison, caption="Left: Original | Right: Reconstructed")
                    )

                # Log images with wandb
                reporter.log_images(
                    wandb_images,
                    prefix=f"reconstructions/epoch_{epoch}"
                )

        # Print epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        # Log epoch metrics
        if reporter:
            reporter.log_epoch(avg_loss=avg_loss, epoch=epoch)

    print("\nTraining complete!")
    return model, losses


def evaluate_autoencoder(model, X_test, batch_size=1000, reporter=None, split="test"):
    """
    Evaluate autoencoder reconstruction quality on test set.

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

    total_error = 0.0
    total_samples = 0

    print(f"\nEvaluating on {n_samples} test samples...")

    for batch_idx in tqdm(range(n_batches), desc="Evaluating"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        X_batch = Value(X_test[start:end])

        # Forward pass
        reconstructed = model(X_batch)

        # Calculate reconstruction error (MSE between original and reconstructed)
        error = np.mean((X_batch.data - reconstructed.data) ** 2)

        # For reconstruction quality, we can use the error as a metric
        # Since there's no classification, we just track the error
        total_error += error * len(X_batch.data)
        total_samples += len(X_batch.data)

    avg_error = total_error / total_samples
    print(f"\n{split.capitalize()} Average Reconstruction Error: {avg_error:.4f}")

    # Log evaluation metrics
    if reporter:
        reporter.log_evaluation(reconstruction_error=avg_error, split=split)

    return avg_error


if __name__ == "__main__":
    # Training configuration
    config = {
        "epochs": 60,
        "batch_size": 256,
        "lr": 1e-3,
        "architecture": "MLP",
        "hidden_layers": [128, 64],
        "activation": "relu"
    }

    # Initialize wandb reporter (set enabled=False to disable wandb logging)
    reporter = TrainingReporter(
        project="tiny-micro-torch-mnist-autoencoder",
        name="mnist-autoencoder-example",
        config=config,
        tags=["mnist", "autoencoder", "example"],
        enabled=True  # Set to False to disable wandb
    )

    # Train the model
    model, losses = train_autoencoder(
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        reporter=reporter
    )

    # Load test data for evaluation
    _, _, X_test, _ = load_mnist()

    # Evaluate on test set
    test_accuracy = evaluate_autoencoder(model, X_test, reporter=reporter, split="test")
    print(f"Final Test Reconstruction Error: {test_accuracy:.4f}")

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
        plt.ylabel('Reconstruction Error')
        plt.title('Autoencoder Training Reconstruction Error Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('autoencoder_training_loss.png')
        print("\nLoss plot saved as 'autoencoder_training_loss.png'")
    except ImportError:
        print("\nMatplotlib/pandas not available - skipping loss plot")

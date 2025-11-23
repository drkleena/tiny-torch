import numpy as np
from tqdm import tqdm
from autograd import Tensor
from nn import Network, Linear, Conv2D, MaxPool2D, Flatten, cross_entropy
from optim import SGD, Adam
from data import load_mnist
from reporters import TrainingReporter

def train_mnist_conv(epochs=20, batch_size=256, lr=0.01, reporter=None):
    """
    Train a simple CNN on MNIST.
    """

    print("Loading MNIST dataset...")
    
    X_train, y_train, X_test, y_test = load_mnist()
    
    X_train = X_train
    y_train = y_train
    X_test  = X_test
    y_test  = y_test

    # X_* currently (N, 784). Reshape to (N, 1, 28, 28)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test  = X_test.reshape(-1, 1, 28, 28)

    print("\nBuilding conv model...")

    model = Network([
        Conv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, activation="relu"),
        MaxPool2D(pool_size=2, stride=1),   # 28x28 -> 14x14
        Flatten(),                           # 16 * 7 * 7 = 784
        Linear(1875, 64, activation="relu"),  # ~50k params instead of ~700k
        Linear(64, 10, activation=None),
    ])

    optimizer = Adam(model.parameters(), lr=lr)

    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    losses = []

    print(f"\nTraining on {n_samples} samples for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Batches per epoch: {n_batches}")
    print(f"Learning rate: {lr}\n")

    for epoch in range(epochs):
        # Shuffle
        permutation = np.random.permutation(n_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_losses = []

        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            X_batch_np = X_train_shuffled[start:end]       # (B, C=1, 28, 28)
            y_batch_np = y_train_shuffled[start:end]       # (B, 10) one-hot

            X_batch = Tensor(X_batch_np)
            y_batch = Tensor(y_batch_np)

            logits = model(X_batch)    # (B, 10)
            loss = cross_entropy(logits, y_batch)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_loss = float(loss.data)
            epoch_losses.append(batch_loss)
            losses.append(batch_loss)

            if reporter:
                reporter.log_batch(
                    loss=batch_loss,
                    batch_idx=batch_idx,
                    epoch=epoch
                )
        
        avg_loss = np.mean(epoch_losses)
        
        val_accuracy = evaluate_mnist_conv(model, X_test, y_test, batch_size, reporter=reporter)
        train_accuracy = evaluate_mnist_conv(model, X_train, y_train, batch_size, reporter=reporter, split="train")
        
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {val_accuracy:.4f}")
        print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_accuracy:.4f}")

        if reporter:
            reporter.log_epoch(avg_loss=avg_loss, epoch=epoch)

    print("\nConv training complete!")
    return model, losses

def evaluate_mnist_conv(model, X_test, y_test, batch_size=512, reporter=None, split="test"):
    """
    Evaluate CNN on MNIST test set.
    """
    n_samples = X_test.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    correct = 0
    total = 0

    print(f"\nEvaluating on {n_samples} test samples...")

    for batch_idx in tqdm(range(n_batches), desc="Evaluating"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        X_batch_np = X_test[start:end].reshape(-1, 1, 28, 28)
        y_batch_np = y_test[start:end]

        X_batch = Tensor(X_batch_np)
        # y_batch stays as raw numpy for argmax
        logits = model(X_batch)

        predictions = np.argmax(logits.data, axis=1)
        targets = np.argmax(y_batch_np, axis=1)

        correct += np.sum(predictions == targets)
        total += len(predictions)

    accuracy = correct / total
    print(f"\n{split.capitalize()} Accuracy: {accuracy:.4f} ({correct}/{total} correct)")

    if reporter:
        reporter.log_evaluation(accuracy=accuracy, split=split)

    return accuracy

if __name__ == "__main__":
    config = {
        "epochs": 60,
        "batch_size": 50,
        "lr": 0.005,
        "architecture": "ConvNet",
        "hidden_layers": [64],
        "activation": "relu"
    }

    reporter = TrainingReporter(
        project="tiny-micro-torch-mnist",
        name="mnist-conv-baseline-adam-maxpool-test",
        config=config,
        tags=["mnist", "cnn", "baseline", "adam", "maxpool-test"],
        enabled=True,
    )

    model, losses = train_mnist_conv(
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        reporter=reporter
    )

    reporter.finish()

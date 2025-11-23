"""
Keras benchmark for MNIST ConvNet to compare against tiny-micro-torch implementation.
This uses the same architecture as mnist_conv.py for fair comparison.
"""
import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import layers


def load_mnist_keras():
    """Load MNIST data in the same format as the custom loader."""
    # Load MNIST from Keras
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape to (N, 28, 28, 1) - Keras uses channels-last
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test


def build_model():
    """
    Build a model matching the tiny-micro-torch architecture:
    - Conv2D(1->3, 3x3, stride=1, padding=0)
    - MaxPool2D(pool_size=2, stride=1)
    - Flatten
    - Dense(1875->64)
    - Dense(64->10)
    """
    model = keras.Sequential([
        # Conv block
        layers.Conv2D(
            3,
            kernel_size=3,
            strides=1,
            padding='valid',  # padding=0
            activation='relu',
            input_shape=(28, 28, 1)
        ),  # 28x28 -> 26x26
        layers.MaxPooling2D(
            pool_size=2,
            strides=1,
            padding='valid'
        ),  # 26x26 -> 25x25

        # Fully connected layers
        layers.Flatten(),  # 25*25*3 = 1875
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # softmax for categorical_crossentropy
    ])

    return model


def train_benchmark(epochs=60, batch_size=50, lr=0.05):
    """Train the Keras model with same hyperparameters."""

    print("="*60)
    print("KERAS BENCHMARK FOR MNIST CONVNET")
    print("="*60)

    # Load data
    print("\nLoading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_keras()

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")

    # Build model
    print("\nBuilding model...")
    model = build_model()

    # Display architecture
    model.summary()

    # Compile with Adam optimizer matching the custom implementation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training callbacks
    callbacks = [
        keras.callbacks.History(),
    ]

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Optimizer: Adam")
    print()

    # Train
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=callbacks
    )

    training_time = time.time() - start_time

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Average time per epoch: {training_time/epochs:.2f} seconds")
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Print epoch-by-epoch comparison
    print("\n" + "="*60)
    print("EPOCH-BY-EPOCH RESULTS")
    print("="*60)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-"*60)

    for i in range(epochs):
        print(f"{i+1:<6} {history.history['loss'][i]:<12.4f} "
              f"{history.history['accuracy'][i]:<12.4f} "
              f"{history.history['val_loss'][i]:<12.4f} "
              f"{history.history['val_accuracy'][i]:<12.4f}")

    return model, history


if __name__ == "__main__":
    # Match the hyperparameters from mnist_conv.py
    model, history = train_benchmark(
        epochs=60,
        batch_size=50,
        lr=0.005
    )

    print("\n" + "="*60)
    print("Benchmark complete! Compare these results with your")
    print("tiny-micro-torch implementation in mnist_conv.py")
    print("="*60)

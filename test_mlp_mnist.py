import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from section.deep_learning.DLBlocks import (
    Sequential,
    MyFullConnectedBlock,
    ReLU
)


def one_hot(y, num_classes=10):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


def load_mnist():
    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten images: (N, 28, 28) -> (N, 784)
    X_train = X_train.reshape(-1, 784).astype(np.float32)
    X_test = X_test.reshape(-1, 784).astype(np.float32)

    # Normalize pixel values
    X_train /= 255.0
    X_test /= 255.0

    # One-hot encoding
    Y_train = one_hot(y_train)
    Y_test = one_hot(y_test)

    return X_train, Y_train, X_test, Y_test


def show_wrong_predictions(model, X, Y, num_images=10):
    predictions = model.predict(X)
    true_labels = np.argmax(Y, axis=1)

    wrong_indices = np.where(predictions != true_labels)[0]
    wrong_indices = wrong_indices[:num_images]

    if len(wrong_indices) == 0:
        print("No incorrect predictions!")
        return

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, wrong_indices):
        ax.imshow(X[idx].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"True: {true_labels[idx]} | Pred: {predictions[idx]}"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = load_mnist()

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    model = Sequential(
        layers=[
            MyFullConnectedBlock((784, 128)),
            ReLU(),
            MyFullConnectedBlock((128, 64)),
            ReLU(),
            MyFullConnectedBlock((64, 10))
        ],
        batch_size=32,
        lr=0.01,
        epochs=15
    )

    # Training
    model.fit(X_train, Y_train)

    # Evaluation
    print("\nTrain evaluation:")
    model.evaluate(X_train, Y_train)

    print("\nTest evaluation:")
    model.evaluate(X_test, Y_test)

    # Plot loss
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    # Show incorrect predictions
    show_wrong_predictions(model, X_test, Y_test)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from neural_network import (
    Sequential,
    MyConvolution,
    MyPooling,
    Flatten,
    MyFullConnectedBlock,
    ReLU
)


def one_hot(y, num_classes=10):
    Y = np.zeros((len(y), num_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1
    return Y


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Add channel dimension: (N, 1, 28, 28)
    X_train = X_train[:, None, :, :]
    X_test = X_test[:, None, :, :]

    Y_train = one_hot(y_train)
    Y_test = one_hot(y_test)

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":

    # 1. Load data
    X_train, Y_train, X_test, Y_test = load_mnist()

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    # 2. Build CNN
    model = Sequential(
        layers=[
            MyConvolution((6, 1, 5, 5)),
            ReLU(),
            MyPooling((2, 2), stride=2),

            Flatten(),

            MyFullConnectedBlock((864, 128)),
            ReLU(),

            MyFullConnectedBlock((128, 10))
        ],
        batch_size=64,
        lr=0.01,
        epochs=10
    )

    # 3. Training
    model.fit(X_train, Y_train)

    # 4. Evaluation
    print("\nTrain evaluation:")
    model.evaluate(X_train, Y_train)

    print("\nTest evaluation:")
    model.evaluate(X_test, Y_test)

    # 5. Plot loss
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training Loss")
    plt.grid()
    plt.show()

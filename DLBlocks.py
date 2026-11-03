import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


class ShapeError(Exception):
    pass

def softmax(X):
    X_max = np.max(X,axis=1,keepdims=True)
    X = X-X_max
    exp_X = np.exp(X)
    return exp_X/np.sum(exp_X,axis=1,keepdims=True)
class Layer(ABC):
    @abstractmethod
    def forward_calculate(self,X) :
        pass

    @abstractmethod
    def backward_calculate(self,gradY):
        pass
class UpdatableLayer(Layer):
    @abstractmethod
    def update_parameter(self,lr):
        pass
class ReLU(Layer):
    def forward_calculate(self, Y):
        self.Y = Y
        return np.maximum(Y, 0)

    def backward_calculate(self, gradY):
        return gradY * (self.Y > 0)


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward_calculate(self, Y):
        self.Y = Y
        return np.where(Y > 0, Y, self.alpha * Y)

    def backward_calculate(self, gradY):
        grad = np.where(self.Y > 0, 1.0, self.alpha)
        return gradY * grad


class Sigmoid(Layer):
    def forward_calculate(self, Y):
        # Clip de han che overflow khi exp
        Y = np.clip(Y, -500, 500)
        self.output = 1 / (1 + np.exp(-Y))
        return self.output

    def backward_calculate(self, gradY):
        return gradY * self.output * (1 - self.output)


class Tanh(Layer):
    def forward_calculate(self, Y):
        self.output = np.tanh(Y)
        return self.output

    def backward_calculate(self, gradY):
        return gradY * (1 - self.output ** 2)


class Identity(Layer):
    def forward_calculate(self, Y):
        return Y

    def backward_calculate(self, gradY):
        return gradY

class MyConvolution(UpdatableLayer):

    def __init__(self,shapeCov: tuple):
        # shapeCov  = (cha_out,cha_in,ph,pw)
        self.shapeCov = shapeCov
        self.cha_in = shapeCov[1]
        self.cha_out = shapeCov[0]

        self.W = np.random.rand(*shapeCov)*0.01# unpack
        self.B = np.zeros(self.cha_out)
        self.gradW = np.zeros(shapeCov)
        self.gradB = np.zeros(self.cha_out)

    from numpy.lib.stride_tricks import sliding_window_view

    def forward_calculate(self, X: np.ndarray):
        self.X = X
        ph, pw = self.shapeCov[2], self.shapeCov[3]

        # windows: (N, C_in, H_out, W_out, ph, pw)
        windows = sliding_window_view(X, (ph, pw), axis=(2, 3))

        # W: (C_out, C_in, ph, pw)
        Y = np.einsum('nchwij,ocij->nohw', windows, self.W, optimize=True)
        Y += self.B[None, :, None, None]
        return Y

    def backward_calculate(self, gradY: np.ndarray):
        ph, pw = self.shapeCov[2], self.shapeCov[3]

        windows = sliding_window_view(self.X, (ph, pw), axis=(2, 3))

        # gradW: (C_out, C_in, ph, pw)
        self.gradW = np.einsum(
            'nohw,nchwij->ocij',
            gradY, windows,
            optimize=True
        )

        # gradB: (C_out,)
        self.gradB[:] = gradY.sum(axis=(0, 2, 3))

        # gradX: lặp theo vị trí trong kernel (chỉ ph*pw vòng)
        gradX = np.zeros_like(self.X)
        H_out, W_out = gradY.shape[2], gradY.shape[3]
        for i in range(ph):
            for j in range(pw):
                # (N, C_out, H_out, W_out) x (C_out, C_in) -> (N, C_in, H_out, W_out)
                gradX[:, :, i:i + H_out, j:j + W_out] += np.einsum(
                    'nohw,oc->nchw', gradY, self.W[:, :, i, j], optimize=True
                )
        return gradX
    def update_parameter(self,lr):
        self.B = self.B - self.gradB*lr
        self.W = self.W - self.gradW*lr



class MyPooling(Layer):
    def __init__(self, shapePooling: tuple, stride=None):
        # shapePooling = (ph, pw)
        # stride = (sh, sw)
        self.shapePooling = shapePooling
        self.stride = stride if stride is not None else shapePooling

        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        self.shapeX = None
        self.argmax = None

    def backward_calculate(self, gradY):
        # gradY: (B, C, out_h, out_w)
        B, C, H, W = self.shapeX
        ph, pw = self.shapePooling
        sh, sw = self.stride

        out_h, out_w = gradY.shape[2:]

        gradX = np.zeros(self.shapeX, dtype=gradY.dtype)

        for ih in range(ph):
            for iw in range(pw):
                k = ih * pw + iw

                mask = (self.argmax == k)

                gradX[
                    :, :,
                    ih:ih + sh * out_h:sh,
                    iw:iw + sw * out_w:sw
                ] += gradY * mask

        return gradX

    def forward_calculate(self, X):
        # X: (B, C, H, W)
        self.shapeX = X.shape

        ph, pw = self.shapePooling
        sh, sw = self.stride

        windows = sliding_window_view(
            X, (ph, pw), axis=(2, 3)
        )
        windows = windows[:, :, ::sh, ::sw, :, :]

        self.argmax = windows.reshape(
            *windows.shape[:4], ph * pw
        ).argmax(axis=-1)

        return windows.max(axis=(-2, -1))

class Flatten(Layer):
    def forward_calculate(self, X: np.ndarray):
        self.shapeX = X.shape
        return X.reshape(X.shape[0], -1)

    def backward_calculate(self, gradX):
        return gradX.reshape(self.shapeX)
class MyFullConnectedBlock(UpdatableLayer):
    def __init__(self, shapeW):
        # shapeW = (input_size, output_size)
        # X: (batch, input_size)
        # Y: (batch, output_size)

        self.shapeW = (shapeW[1], shapeW[0])

        self.X = None
        self.W = np.random.randn(*self.shapeW) * 0.01
        self.B = np.zeros(self.shapeW[0])

        self.gradW = np.zeros(self.shapeW)
        self.gradB = np.zeros(self.shapeW[0])

    def forward_calculate(self, X):
        self.X = X
        return (self.W @ X.T).T + self.B

    def backward_calculate(self, gradY):
        # gradY: (batch, output_size)

        self.gradB = np.sum(gradY, axis=0)
        self.gradW = gradY.T @ self.X

        gradX = gradY @ self.W

        return gradX

    def update_parameter(self, lr):
        self.W -= lr * self.gradW
        self.B -= lr * self.gradB
class Sequential:

    def __init__(
        self,
        layers: list[Layer],
        batch_size: int = 10,
        lr: float = 0.01,
        loss_func_type: str = 'cross-entropy',
        epochs: int = 1
    ):
        if not layers:
            raise ValueError("Model must contain at least one layer.")

        if batch_size <= 0 or epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")

        if lr <= 0:
            raise ValueError("Learning rate must be positive.")

        self.layers = layers
        self.batch_size = batch_size
        self.lr = lr
        self.loss_func_type = loss_func_type
        self.epochs = epochs

        self.loss_history = []

    def forward_calculate(self, X):
        Z = X

        for layer in self.layers:
            Z = layer.forward_calculate(Z)

        return Z

    def backward_calculate(self, gradY):
        gradZ = gradY

        for layer in reversed(self.layers):
            gradZ = layer.backward_calculate(gradZ)

        return gradZ

    def loss(self, Y_hat, Y):
        if self.loss_func_type == 'cross-entropy':
            possibilities = softmax(Y_hat)
            possibilities = np.clip(possibilities, 1e-15, 1.0)

            return -np.sum(Y * np.log(possibilities)) / Y.shape[0]

        raise NotImplementedError(
            f"Loss function {self.loss_func_type} is not implemented."
        )

    def calculate_gradY(self, Y_hat, Y):
        # Mean gradient theo batch
        return (softmax(Y_hat) - Y) / Y.shape[0]

    def fit(self, X, Y):
        N = X.shape[0]

        if N == 0 or N != Y.shape[0]:
            raise ValueError("Invalid training data.")

        if self.loss_func_type == 'cross-entropy':
            if Y.ndim != 2:
                raise ValueError("Cross-entropy requires one-hot labels.")

        for ep in range(self.epochs):
            # Shuffle dữ liệu mỗi epoch
            indices = np.random.permutation(N)
            X = X[indices]
            Y = Y[indices]

            total_loss = 0.0

            for i in range(0, N, self.batch_size):
                Xb = X[i:i + self.batch_size]
                Yb = Y[i:i + self.batch_size]

                # Forward
                Y_hat = self.forward_calculate(Xb)

                # Loss
                batch_loss = self.loss(Y_hat, Yb)
                total_loss += batch_loss * len(Xb)

                # Backward
                gradY = self.calculate_gradY(Y_hat, Yb)
                self.backward_calculate(gradY)

                # Update parameters
                for layer in self.layers:
                    if isinstance(layer, UpdatableLayer):
                        layer.update_parameter(self.lr)

            avg_loss = total_loss / N
            self.loss_history.append(avg_loss)

            print(
                f"Epoch {ep + 1}/{self.epochs} - "
                f"Loss: {avg_loss:.6f}"
            )

        return self

    def predict(self, X):
        Y_hat = self.forward_calculate(X)

        if self.loss_func_type == 'cross-entropy':
            return np.argmax(softmax(Y_hat), axis=1)

        return Y_hat

    def evaluate(self, X, Y):
        Y_pred = self.predict(X)

        if self.loss_func_type == 'cross-entropy':
            Y_true = np.argmax(Y, axis=1) if Y.ndim > 1 else Y

            accuracy = np.mean(Y_pred == Y_true)

            print(
                f"Accuracy: {accuracy:.4f} "
                f"({accuracy * 100:.2f}%)"
            )

            return accuracy

        mse = np.mean((Y_pred - Y) ** 2)
        mae = np.mean(np.abs(Y_pred - Y))

        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")

        return mse, mae
def print_distribution(Y_train, Y_test):
    y_train = np.argmax(Y_train, axis=1) if Y_train.ndim > 1 else Y_train
    y_test = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test

    print("Train set:")
    for c in np.unique(np.concatenate([y_train, y_test])):
        count = np.sum(y_train == c)
        print(f"Class {c}: {count} ({count / len(y_train) * 100:.2f}%)")

    print("\nTest set:")
    for c in np.unique(np.concatenate([y_train, y_test])):
        count = np.sum(y_test == c)
        print(f"Class {c}: {count} ({count / len(y_test) * 100:.2f}%)")
def show_image(inputX):
    X = np.asarray(inputX)

    # Nếu có batch dimension, lấy ảnh đầu tiên
    if X.ndim == 4:
        X = X[0]
    elif X.ndim == 3:
        # Phân biệt batch ảnh grayscale với ảnh nhiều channel
        if X.shape[0] not in (1, 3, 4) and X.shape[-1] not in (1, 3, 4):
            X = X[0]

    # Hiển thị ảnh 2D
    if X.ndim == 2:
        plt.imshow(X, cmap="gray")

    elif X.ndim == 3:
        # Channel-first: (C, H, W) -> (H, W, C)
        if X.shape[0] in (1, 3, 4):
            X = np.transpose(X, (1, 2, 0))

        # Grayscale
        if X.shape[-1] == 1:
            plt.imshow(X.squeeze(-1), cmap="gray")
        # RGB hoặc RGBA
        elif X.shape[-1] in (3, 4):
            plt.imshow(X)
        else:
            raise ValueError(f"Không hỗ trợ shape: {X.shape}")
    else:
        raise ValueError(f"Không hỗ trợ shape: {X.shape}")

    plt.axis("off")
    plt.show()
model = Sequential(
    layers=[
        MyFullConnectedBlock((784, 128)),
        ReLU(),
        MyFullConnectedBlock((128, 64)),
        ReLU(),
        MyFullConnectedBlock((64, 10))
    ],
    batch_size=64,
    lr=0.01,
    epochs=5
)


# DL Blocks

A lightweight deep learning framework built from scratch using **NumPy**. This project implements the core components of neural networks, including forward propagation, backpropagation, gradient computation, and parameter updates.

The main goal is to understand how deep learning frameworks work internally by implementing their fundamental operations without relying on high-level deep learning libraries.

## Features

### Layers

- **Fully Connected Layer** (`MyFullConnectedBlock`)
- **Convolutional Layer** (`MyConvolution`)
- **Max Pooling Layer** (`MyPooling`)
- **Flatten Layer** (`Flatten`)

### Activation Functions

- ReLU
- Leaky ReLU
- Sigmoid
- Tanh
- Identity

### Model

- Sequential architecture
- Mini-batch training
- Cross-entropy loss
- Softmax output
- Backpropagation
- Gradient-based parameter updates
- Model evaluation

## Implementation Details

The framework uses NumPy to implement mathematical operations directly.

Some key concepts explored in this project include:

- Matrix multiplication and tensor operations
- Convolution using sliding windows and `einsum`
- Gradient computation through backpropagation
- Numerical gradient checking
- Mini-batch gradient descent

## Experiments

The framework has been tested on the MNIST handwritten digit dataset.

### 1. Multilayer Perceptron (MLP)

Architecture:

```text
Input (784)
    |
Fully Connected (128)
    |
ReLU
    |
Fully Connected (64)
    |
ReLU
    |
Fully Connected (10)
    |
Softmax
```

| Metric | Result |
|---|---:|
| Epochs | 15 |
| Batch Size | 64 |
| Learning Rate | 0.01 |
| Training Accuracy | 92.08% |
| Testing Accuracy | 92.00% |

### 2. Convolutional Neural Network (CNN)

Architecture:

```text
Input (1 × 28 × 28)
    |
Convolution (6 filters, 5 × 5)
    |
ReLU
    |
Max Pooling (2 × 2)
    |
Flatten
    |
Fully Connected (128)
    |
ReLU
    |
Fully Connected (10)
    |
Softmax
```

| Metric | Result |
|---|---:|
| Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.01 |
| Training Accuracy | 95.66% |
| Testing Accuracy | 95.85% |

## Gradient Checking

Numerical gradient checking was performed to verify the correctness of the implemented layers.

| Layer | Relative Error |
|---|---:|
| Fully Connected | ~\(10^{-14}\) |
| Convolution | ~\(10^{-11}\) |

These results indicate that the analytical gradients closely match numerical gradients in the tested cases.

## Requirements

- Python 3
- NumPy
- Matplotlib
- TensorFlow (only for downloading the MNIST dataset)

Install dependencies:

```bash
pip install numpy matplotlib tensorflow
```

## Getting Started

Clone the repository:

```bash
git clone <repository-url>
cd DL-Blocks
```

Run an experiment:

```bash
python test_mlp_mnist.py
```

or

```bash
python test_cnn_mnist.py
```

## Project Purpose

This project is primarily an educational implementation designed to explore the internal mechanisms of deep learning, rather than to replace production frameworks such as PyTorch or TensorFlow.
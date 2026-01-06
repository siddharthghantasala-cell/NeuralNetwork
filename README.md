# Neural Network

🚧 **Work in Progress** — Core functionality works, training loop and testing in development.

A from-scratch implementation of a feedforward neural network with backpropagation, built using only NumPy. Created as a learning project to deeply understand the mathematics behind neural networks.

## Overview

This project implements a fully-connected neural network that learns through backpropagation. Rather than using a deep learning framework, everything is built from first principles to understand how the elegant linear algebra of backpropagation actually works:

- Error propagates backward through layers via matrix-transpose multiplication
- Weight gradients are computed as outer products, perfectly matching the weight matrix dimensions
- The chain rule flows naturally through the network architecture

## Features

- **Custom Layer Implementation**: Each layer handles its own forward pass, weight management, and backpropagation
- **Flexible Architecture**: Configurable input size, output size, number of hidden layers, and hidden layer dimensions
- **Activation Functions**: Support for multiple activation functions (ReLU, etc.) with derivative calculations
- **Backpropagation**: Full implementation of the backpropagation algorithm for training
- **Gradient Descent**: Weight and bias optimization with configurable learning rate
- **Minimal Dependencies**: NumPy only, two-file implementation

## Project Structure

- **Layer**: Core building block managing inputs, weights, biases, and computations for a single layer
- **Network**: Orchestrates multiple layers, handling forward pass and backpropagation across the entire network
- **ActivationFunctions**: Module containing activation functions (ReLU, sigmoid, etc.) and their derivatives

## How It Works

### Forward Pass

Input propagates through each layer sequentially:

1. Input is passed to the first layer
2. Each layer computes: `output = activation(weights @ input + biases)`
3. Output becomes input for the next layer
4. Final output is produced by the output layer

### Backpropagation

The network learns by computing gradients and updating weights:

1. **Output error**: Calculate error at the output layer using the cost function derivative (MSE)
2. **Error propagation**: For each layer going backward, compute the error term δ by multiplying the transposed weight matrix with the next layer's error, then element-wise multiply by the activation derivative
3. **Weight gradients**: Compute gradients as the outer product of the error and the previous layer's activations — this produces a matrix exactly matching the weights' dimensions
4. **Update**: Adjust weights and biases using the gradients scaled by learning rate

## Usage

```python
import numpy as np
from NeuralNetwork import Network

# Create sample input data
input_data = np.random.rand(2, 1)

# Initialize network
# Parameters: input_size, output_size, hidden_layer_count, hidden_layer_size, inputs
network = Network(
    input_size=2,
    output_size=2,
    hidden_layer_count=1,
    hidden_layer_size=3,
    inputs=input_data
)

# Forward pass
network.forward()
network.show_output()

# Single backpropagation step
expected_output = np.array([[0.5], [0.8]])
learning_rate = 0.01
network.backpropagation(expected_output, learning_rate)

# Training loop (basic example)
for epoch in range(1000):
    network.forward()
    network.backpropagation(expected_output, learning_rate)
```

## Network Parameters

| Parameter | Description |
|-----------|-------------|
| `input_size` | Dimension of input vectors |
| `output_size` | Dimension of output vectors |
| `hidden_layer_count` | Number of hidden layers (0 = direct input-to-output) |
| `hidden_layer_size` | Number of neurons in each hidden layer |
| `inputs` | Initial input data (numpy array or list) |
| `learning_rate` | Weight adjustment scale during training (typical: 0.001–0.1) |

## Key Methods

### Layer

| Method | Description |
|--------|-------------|
| `forward()` | Performs forward pass computation |
| `set_inputs(inputs)` | Sets input values for the layer |
| `backprop(diff, learning_rate)` | Computes gradients and updates weights |

### Network

| Method | Description |
|--------|-------------|
| `forward()` | Executes full forward pass through all layers |
| `show_output()` | Displays final network output |
| `backpropagation(d, learning_rate)` | Performs complete backpropagation with target values `d` |

## Status & Roadmap

### Currently Implementing
- [ ] Complete training loop with batch processing
- [ ] Cross-entropy loss function
- [ ] Test suite
- [ ] Example implementations (MNIST, XOR, etc.)

### Planned Enhancements
- [ ] Additional loss functions (binary cross-entropy, custom)
- [ ] Momentum and advanced optimizers (Adam, RMSprop)
- [ ] Regularization (L1, L2)
- [ ] Model saving/loading
- [ ] Training visualization
- [ ] Additional activation functions (sigmoid, tanh, softmax)

## References & Attribution

This implementation was built with reference to Michael Nielsen's excellent book *Neural Networks and Deep Learning*, particularly [Chapter 2: How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html).

The mathematical foundations and intuitions from that chapter were invaluable for understanding and implementing backpropagation from scratch.

## License

This project is open source and available for educational purposes.

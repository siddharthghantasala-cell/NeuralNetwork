# Neural Network

A from-scratch implementation of a feedforward neural network with backpropagation, built using only NumPy. Created as a learning project to deeply understand the mathematics behind neural networks — and grown into a small, fully customizable, general-purpose framework.

## Overview

This project implements a fully-connected neural network that learns through backpropagation. Rather than using a deep learning framework, everything is built from first principles to understand how the elegant linear algebra of backpropagation actually works:

- Error propagates backward through layers via matrix-transpose multiplication
- Weight gradients are computed as batched matrix products, perfectly matching the weight matrix dimensions
- The chain rule flows naturally through the network architecture
- Activations and losses are fully decoupled, so any activation composes correctly with any loss

It is general-purpose: you choose the architecture, the activation functions (per network), the loss function, the weight initialization, and the batch size. MNIST and XOR are included as example use cases, but nothing in the core is specialized to either.

## Customizability

The central design goal of this project is that **every meaningful component is pluggable**, with no hidden assumptions baked into the core:

- **Architecture** — arbitrary input size, output size, number of hidden layers, and hidden layer width.
- **Activation functions** — sigmoid, ReLU, tanh, and softmax, each passed in as a function. The hidden layers and the output layer can use different activations.
- **Loss functions** — MSE and cross-entropy, passed into the training method.
- **Initialization** — pluggable initialization strategies (e.g. He initialization), specified separately for hidden layers and the output layer.

The piece that makes this work cleanly is that **activations and losses know nothing about each other.** Each loss computes only its own gradient with respect to the network's output. Each activation computes only its own Jacobian-vector product against the upstream gradient. The output layer simply composes the two. For element-wise activations (sigmoid, ReLU, tanh) the Jacobian is diagonal, so this reduces to the familiar Hadamard product. For softmax — whose outputs all depend on every input — the activation returns the full Jacobian-vector product `s * (upstream - Σ(s * upstream))`.

Because of this decoupling, the famous softmax + cross-entropy gradient (`predicted - actual`) is **never hardcoded** — it emerges automatically from composing softmax's backward with cross-entropy's derivative. The same machinery also handles non-standard pairings (e.g. softmax + MSE) correctly, rather than assuming the common combination.

## Features

- **Custom Layer Implementation**: Each layer handles its own forward pass, weight management, gradient accumulation, and weight updates
- **Flexible Architecture**: Configurable input size, output size, number of hidden layers, and hidden layer dimensions
- **Pluggable Activation Functions**: ReLU, sigmoid, tanh, and softmax, each with proper derivative / Jacobian-vector handling
- **Pluggable Loss Functions**: Mean Squared Error and Cross-Entropy, fully decoupled from the activations
- **Pluggable Initialization**: He initialization included, with support for custom initialization strategies
- **Vectorized Mini-Batch Gradient Descent**: Whole batches are processed as matrix operations; gradients are accumulated and averaged per batch before a single weight update
- **Separated Compute and Apply**: Gradients are computed during the backward pass and applied in a distinct update step, with explicit gradient resetting between batches
- **Model Saving / Loading**: Trained networks can be serialized and reloaded
- **Training Visualization**: Loss-over-time plotting
- **Minimal Dependencies**: NumPy for the core, Matplotlib for visualization

## Project Structure

- **Layer**: Core building block managing inputs, weights, biases, gradient accumulators, and computations for a single layer
- **Network**: Orchestrates multiple layers, handling the forward pass, backward pass, gradient updates, and the mini-batch training loop
- **ExternalFunctions**: Module containing activation functions (ReLU, sigmoid, tanh, softmax), loss functions (MSE, cross-entropy), and initialization strategies (He), all with their derivatives where applicable

## How It Works

### Forward Pass

Input propagates through each layer sequentially:

1. Input is passed to the first layer (shaped `features × batch` so a whole batch flows through at once)
2. Each layer computes: `output = activation(weights @ input + biases)`
3. Output becomes input for the next layer
4. Final output is produced by the output layer

### Backpropagation

The network learns by computing gradients and updating weights:

1. **Output gradient**: The chosen loss function computes its gradient with respect to the network's output
2. **Error propagation**: For each layer going backward, the activation turns the upstream gradient into the layer's error term (a Hadamard product for element-wise activations, a Jacobian-vector product for softmax), and the error is passed further back by multiplying with the transposed weight matrix
3. **Weight gradients**: Computed as a batched matrix product of the error and the layer's inputs — this sums over the batch automatically and produces a matrix exactly matching the weights' dimensions
4. **Accumulate, average, update**: Per-layer gradients are accumulated across the batch, averaged, and then applied in a single update step; accumulators are reset before the next batch

### Mini-Batch Training

One epoch is a full pass over the data, split into batches:

1. Shuffle the data indices (re-shuffled every epoch)
2. Split into batches of the configured size
3. For each batch: forward pass on the whole batch, compute the loss gradient, backward pass to accumulate gradients, average and apply a single update, then reset the accumulators

## Network Parameters

| Parameter | Description |
|-----------|-------------|
| `input_size` | Dimension of input vectors |
| `output_size` | Dimension of output vectors |
| `hidden_layer_count` | Number of hidden layers (0 = direct input-to-output) |
| `hidden_layer_size` | Number of neurons in each hidden layer |
| `activation_function` | Activation used by the hidden layers |
| `output_activation` | Activation used by the output layer |
| `initialization` | Weight initialization strategy for hidden layers (e.g. `he_initialization`) |
| `output_initialization` | Weight initialization strategy for the output layer |

## Training Parameters

| Parameter | Description |
|-----------|-------------|
| `learning_rate` | Weight adjustment scale during training (typical: 0.001–0.1) |
| `data` | Training inputs |
| `labels` | Training labels |
| `epochs` | Number of full passes over the training data |
| `batch_size` | Number of datapoints per mini-batch |
| `loss` | Loss function used for training (e.g. `mse`, `cross_entropy`) |

## Key Methods

### Layer

| Method | Description |
|--------|-------------|
| `forward(...)` | Performs the forward pass for the layer |
| `set_inputs(inputs)` | Sets input values for the layer |
| `backward(upstream_gradient)` | Computes the layer's error and accumulates weight/bias gradients |
| `update(learning_rate, batch_size)` | Averages the accumulated gradients and applies the weight/bias update |
| `reset_grad()` | Resets the gradient accumulators between batches |

### Network

| Method | Description |
|--------|-------------|
| `forward(inputs)` | Executes a full forward pass through all layers |
| `backward(loss_error, batch_size)` | Propagates the loss gradient backward through every layer |
| `update(learning_rate, batch_size)` | Applies the averaged gradient update across all layers |
| `reset_grad()` | Resets gradient accumulators across the whole network |
| `mini_batch_grad_desc(...)` | Runs full mini-batch gradient descent training |
| `singleton_grad_desc(...)` | Convenience wrapper for online (batch size 1) training |
| `show_output()` | Displays the final network output |
| `return_output()` | Returns the final network output |
| `plot_loss()` | Plots the loss over the course of training |

## Activation & Loss Functions

| Function | Type | Notes |
|----------|------|-------|
| `relu` | Activation | Pairs with He initialization; element-wise |
| `sigmoid` | Activation | Element-wise; useful for output layers in `[0, 1]` |
| `tanh` | Activation | Element-wise; output range `[-1, 1]` |
| `softmax` | Activation | Non-element-wise; returns a Jacobian-vector product in the backward pass |
| `mse` | Loss | Mean Squared Error |
| `cross_entropy` | Loss | Pairs naturally with softmax for classification |
| `he_initialization` | Initialization | Scales weights by `sqrt(2 / fan_in)`, tuned for ReLU |

## Example Results

- **MNIST** (3 hidden layers of 100 units, ReLU + softmax, cross-entropy, He init): **~97.6% test accuracy** on the held-out test set
- **XOR** (1 hidden layer of 3 units, tanh): learns the function to clean separation

## Status & Roadmap

The core network, training loop, vectorized mini-batch gradient descent, multiple activations and losses, initialization strategies, model saving/loading, and training visualization are all complete and working.

### Planned Enhancements
- [ ] Additional loss functions (binary cross-entropy, custom)
- [ ] Momentum and advanced optimizers (Adam, RMSprop)
- [ ] Regularization (L1, L2)
- [ ] Learning rate scheduling

## References & Attribution

This implementation was built with reference to Michael Nielsen's excellent book *Neural Networks and Deep Learning*, particularly [Chapter 2: How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html).

Stanford's [CS231n](https://cs231n.github.io/) notes were also a valuable reference for the vectorized backward pass, gradient accumulation, and the staged forward/backward structure.

## License

This project is open source and available for educational purposes.
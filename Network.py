import numpy as np
from ActivationFunctions import *

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.inputs = np.random.randn(input_size, 1) # Vector of dimensions nx1
        self.weights = np.matrix(np.random.rand(output_size, input_size)) # Matrix of dimensions mxn (to pre multiply with input vector)
        self.biases = np.matrix(np.random.rand(output_size, 1)) # Vector of dimensions (to add to output vector)
        self.activation = activation
        self.outputs = None # Output vector

    def forward(self):
        """
        Forward pass for 1 layer
        :return: Sets outputs to the result of calculating the forward pass, which is the multiplication of the weight matrix and input vectors in that order
        """
        self.outputs = self.weights.dot(self.inputs) + self.biases
        self.outputs = self.activation(self.outputs)

    def set_inputs(self, inputs):
        """
        Sets the inputs for the self.inputs field
        :param inputs: The inputs to be added in the form of a list of numpy array
        :return: None. Sets the inputs field appropriately
        """
        if isinstance(inputs, np.ndarray):
            self.inputs = inputs
        elif isinstance(inputs, list):
            self.inputs = np.array(inputs)
        else:
            raise TypeError('Input must be a numpy array or a list')

    def set_weights(self, weights):
        """
        Purely for testing purposes, in order to get a deterministic output
        :param weights: The weights to be inputted
        :return: None. Sets the new values
        """
        if isinstance(weights, np.ndarray):
            self.weights = weights
        elif isinstance(weights, list):
            self.weights = np.array(weights)
        else:
            raise TypeError('Input must be a numpy array or a list')

    def set_biases(self, biases):
        """
        Purely for testing purposes, in order to get a deterministic output
        :param biases: Biases to be inputted
        :return: None. Sets the new values
        """
        if isinstance(biases, np.ndarray):
            self.biases = biases
        elif isinstance(biases, list):
            self.biases = np.array(biases)
        else:
            raise TypeError('Input must be a numpy array or a list')


class Network:
    def __init__(
            self,
            input_size,
            output_size,
            hidden_layer_count,
            hidden_layer_size,
            inputs,
    ):
        if hidden_layer_count == 0:
            hidden_layer_size = output_size

        self.input_layer = Layer(input_size, hidden_layer_size, stepFunc)
        self.input_layer.set_inputs(inputs)

        self.hidden_layers = [Layer(hidden_layer_size, hidden_layer_size, stepFunc) for i in range(hidden_layer_count)]
        self.output_layer = Layer(hidden_layer_size, output_size, stepFunc)

        """Hard coding some values for testing purposes"""
        self.hidden_layers[0].set_weights(np.ones((hidden_layer_size, hidden_layer_size)))
        self.hidden_layers[0].set_biases(np.ones((3,1)))
        self.output_layer.set_weights(np.ones((2,3)))
        self.output_layer.set_biases(np.ones((2,1)))

    def forward(self):
        """
        Forward pass for the network
        :return: None. It calculates the output of the network with the given inputs
        """
        layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        for i in range(0,len(layers) - 1):
            # Basically feed forward of the first layer, then put those outputs as in the inputs of the next layer
            layers[i].forward()
            layers[i+1].set_inputs(layers[i].outputs)
        self.output_layer.forward()

    def show_output(self):
        """
        Shows the final output of the entire network's calculations
        :return: Prints the output layer's outputs
        """
        print('Final output : ', list(self.output_layer.outputs))

if __name__ == '__main__':
    inputs = np.ones((2, 1))
    test = Network(2,2,1,3, inputs)
    test.forward()
    test.show_output()
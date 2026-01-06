import numpy as np
from ActivationFunctions import *

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.inputs = np.zeros(input_size) # Vector of dimensions nx1
        self.weights = np.matrix(np.random.rand(output_size, input_size)) # Matrix of dimensions mxn (to pre multiply with input vector)
        self.biases = np.matrix(np.random.rand(output_size, 1)) # Vector of dimensions (to add to output vector)
        self.activation = activation
        self.outputs = [None, None] # Output vector of dimensions mx1 where the first index has the value
        # after running through activation function and the second index has the value before running through the activaiton function

    def forward(self):
        """
        Forward pass for 1 layer
        :return: Sets outputs to the result of calculating the forward pass, which is the multiplication of the weight matrix and input vectors in that order
        """
        self.outputs[1] = self.weights.dot(self.inputs) + self.biases
        self.outputs[0] = self.activation(self.outputs[1])

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

    def backprop(self, prev_error, learning_rate):
        """
        A method used to find the gradient of the weights and biases in the output layer
        :param prev_error: The error propagated from previous layer
        :param learning_rate: The learning rate at which the network learns
        :return: Finds the final gradient of the loss function
        """
        # Calculate the linear combination going into the activation function for each output node
        if not isinstance(self.outputs[1], np.ndarray):
            raise ValueError('The output layer has not been set yet')

        # We differentiate the activation function (should probably be the same across all layers but I'm just doing it anyway)
        def d_active(x):
            return self.activation(x, True)

        # For now, simply calculate the error
        """
        For this part, we transpose the weights matrix to intuitively reverse the direction of the weights
        """
        t_weights = self.weights.T

        # Error should be the dimensions of the current layer (output layer)
        # (t_weights @ prev_error) converts prev_error from being a vector with output dimensions to
        # a vector that has input dimensions
        error = (t_weights @ prev_error) * d_active(self.outputs[0])

        """
        Updating the weights
        """
        # To update the weights, we need to derive the weighted input with respect to each weight in
        # order to find how much we need to update the weights by

        # This creates a rank 1 matrix that makes sure that all the weights coming from input 'i' are multiplied
        # by i (after multiplying the calculated error appropriately)
        dW = learning_rate * np.outer(error, self.inputs)

        # Finally update the weights
        self.weights -= dW

        return error


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

        self.input_layer = Layer(input_size, hidden_layer_size, relu)
        self.input_layer.set_inputs(inputs)

        self.hidden_layers = [Layer(hidden_layer_size, hidden_layer_size, relu) for _ in range(hidden_layer_count)]
        self.output_layer = Layer(hidden_layer_size, output_size, relu)

        self.network = [self.input_layer] + self.hidden_layers + [self.output_layer]

        """Hard coding some values for testing purposes"""
        # self.input_layer.set_weights(np.ones((hidden_layer_size, input_size)))
        # self.hidden_layers[0].set_weights(np.ones((hidden_layer_size, hidden_layer_size)))
        # self.hidden_layers[0].set_biases(np.ones((3,1)))
        # self.output_layer.set_weights(np.ones((2,3)))
        # self.output_layer.set_biases(np.ones((2,1)))

    def forward(self):
        """
        Forward pass for the network
        :return: None. It calculates the output of the network with the given inputs
        """
        layers = [self.input_layer] + self.hidden_layers + [self.output_layer]
        for i in range(0,len(layers) - 1):
            # Basically feed forward of the first layer, then put those outputs as in the inputs of the next layer
            layers[i].forward()
            layers[i+1].set_inputs(layers[i].outputs[1])
        self.output_layer.forward()

    def show_output(self):
        """
        Shows the final output of the entire network's calculations
        :return: Prints the output layer's outputs
        """
        print('Final output : ', list(self.output_layer.outputs))

    def backpropagation(self, d, learning_rate):
        """
        A method used for a single backwards pass using backpropagation
        :param d: The actual values
        :return: Sets the weights closer to the correct value for optimal classification
        """

        """
        This process will take place in two steps:
            1. Doing backpropagation from the outputs
            2. Doing backpropagation between hidden layers 
        """

        if (not isinstance(self.output_layer.outputs[1], np.ndarray)
                and not isinstance(self.output_layer.outputs[0], np.ndarray)):
            raise ValueError('Forward pass not done yet')

        """
        Step 1
        First we need to do the backpropagation at the output layer
        """
        # This involves differentiating the cost function (in this case we're using MSE)
        dL = (d - self.output_layer.outputs[1])

        # Then differentiating the activation function
        def d_output_active(x):
            return self.output_layer.activation(x, True)

        # The error vector to be propagated through the network is computed as such
        # which should be the size of the output layer
        output_error = dL * d_output_active(self.output_layer.outputs[0])

        """
        Updating the output weights
        """
        dW_output = learning_rate * np.outer(output_error, self.output_layer.inputs)
        self.output_layer.weights -= dW_output

        """
        Step 2
        Then we need to propagate the error back through the network using the newly obtained error term
        """
        network = self.network[:-1]
        next_error = output_error
        # Go through all the layers (except the output layer), find the error and update the weights accordingly
        # using each layer's backprop method
        for layer in network:
            next_error = layer.backprop(next_error, learning_rate)



if __name__ == '__main__':
    input = np.random.rand(2, 1)
    network = Network(2, 2, 1, 3, input)
    network.forward()
    network.show_output()
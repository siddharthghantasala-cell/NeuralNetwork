import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.inputs = np.random.randn(input_size, 1) # Vector of dimensions nx1
        self.weights = np.matrix(np.random.rand(output_size, input_size)) # Matrix of dimensions mxn (to pre multiply with input vector)
        self.biases = np.matrix(np.random.rand(output_size, 1)) # Vector of dimensions (to add to output vector)
        self.outputs = None # Output vector

    def forward(self):
        """
        Forward pass for 1 layer
        :return: Sets outputs to the result of calculating the forward pass, which is the multiplication of the weight matrix and input vectors in that order
        """
        self.outputs = self.weights.dot(self.inputs)

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
        self.weights = weights

    def set_biases(self, biases):
        """
        Purely for testing purposes, in order to get a deterministic output
        :param biases: Biases to be inputted
        :return: None. Sets the new values
        """
        self.biases = biases


class Network:
    def __init__(self, input_size, output_size, hidden_layer_count, hidden_layer_size, inputs):
        if hidden_layer_count == 0:
            hidden_layer_size = output_size
        self.input_layer = Layer(input_size, hidden_layer_size)
        self.input_layer.set_inputs(inputs)
        self.hidden_layers = [Layer(hidden_layer_size, hidden_layer_size) for i in range(hidden_layer_count)]
        self.output_layer = Layer(hidden_layer_size, output_size)

    def forward(self):
        self.input_layer.forward()

        if len(self.hidden_layers) > 0:
            self.hidden_layers[0].set_inputs(self.input_layer.outputs)
            layer = 1
            for layer in range(1, len(self.hidden_layers) - 1):
                self.hidden_layers[layer].forward()
                self.hidden_layers[layer + 1] = self.hidden_layers[layer].outputs

            self.hidden_layers[layer].forward()
            self.output_layer.set_inputs(self.hidden_layers[layer].outputs)
        else:
            self.output_layer.set_inputs(self.input_layer.outputs)

    def show_output(self):
            print('Final output : ', list(self.output_layer.outputs))

if __name__ == '__main__':
    test = Network(3, 4, 0, 0, np.matrix(np.random.rand(3, 1)))
    test.forward()
    print(test.show_output())
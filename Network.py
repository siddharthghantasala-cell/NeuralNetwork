import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.inputs = np.random.randn(output_size, input_size)
        self.weights = np.matrix(np.random.rand(input_size, output_size))
        self.biases = np.matrix(np.random.rand(input_size, output_size))
        self.outputs = None

    def forward(self):
        self.outputs = self.weights.dot(self.inputs)

    def set_inputs(self, inputs):
        if isinstance(inputs, np.ndarray):
            self.inputs = inputs
        elif isinstance(inputs, list):
            self.inputs = np.array(inputs)
        else:
            raise TypeError('Input must be a numpy array or a list')

class Network:
    def __init__(self, input_size, output_size, hidden_layer_count, hidden_layer_size, inputs):
        self.input_layer = Layer(input_size, hidden_layer_size)
        self.hidden_layer = np.array(
            [Layer(hidden_layer_size, hidden_layer_size) for i in range(hidden_layer_count)]
        )
        self.output_layer = Layer(hidden_layer_size, output_size)

    def forward(self):

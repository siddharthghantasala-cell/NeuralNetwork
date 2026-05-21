import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.inputs = np.random.randn(input_size,) # Vector of dimensions nx1
        self.weights = np.random.randn(output_size, input_size) # Matrix of dimensions mxn (to pre multiply with input vector)
        self.biases = np.random.randn(output_size,) # Vector of dimensions (to add to output vector)
        self.activation = activation
        self.outputs = [None, None] # Output vector of dimensions mx1 where the first index has the value
        # after running through activation function and the second index has the value before running through the activation function
        self.dW = np.zeros(output_size, input_size) # This is the accumulated gradient so far by the current batch of training samples waiting to be applied to the weights in
        # the update pass

        self.input_size = input_size
        self.output_size = output_size

    def forward(self):
        """
        Forward pass for 1 layer
        :return: Sets outputs to the result of calculating the forward pass, which is the multiplication of the weight matrix and input vectors in that order
        """
        self.outputs[0] = np.dot(self.weights, self.inputs) + self.biases
        self.outputs[1] = self.activation(self.outputs[0])

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

    def backward(self, upstream_gradient, learning_rate):
        """
        A method used to run a backwards pass on one layer. Calculates upstream gradient and layer gradient
        :param upstream_gradient: The gradient propagated from the previous layer
        :param learning_rate: The learning rate
        :return: Finds the final gradient of the loss function
        """
        # Calculate the linear combination going into the activation function for each output node
        if not isinstance(self.outputs[1], np.ndarray):
            raise ValueError('The output layer has not been set yet')

        # We differentiate the activation function (should probably be the same across all layers, but I'm just doing it anyway)
        def d_active(x):
            return self.activation(x, True)

        # To update the weights, we need to derive the weighted input with respect to each weight in
        # order to find how much we need to update the weights by

        # Error should be the dimensions of the current layer (output layer)
        # upstream gradient multiplied the previous layer's transposed weights with that layer's error
        # thereby converting the output dimensions of that layer to it's input dimensions which
        # also is this layer's output dimensions allowing us to effectively pull the error from the next
        # layer to this layer and do the Hadamard product
        error = upstream_gradient * d_active(self.outputs[0])

        # This creates a rank 1 matrix that makes sure that all the weights coming from input 'i' are multiplied
        # by i (after multiplying the calculated error appropriately) and accumulates all gradients of all data
        # points in the batch
        self.dW += learning_rate * np.outer(error, self.inputs)

        # For now, simply calculate the error
        """
        For this part, we transpose the weights matrix to intuitively reverse the direction of the weights
        """
        t_weights = self.weights.T

        # Record the next upstream gradient for the next layer
        new_gradient = t_weights @ error

        return new_gradient



    def update(self, error, learning_rate, batch_size):
        """
        A method used to only update the weights and biases of the layer during the update pass
        :param error: The error to be propagated
        :param learning_rate: The learning rate
        :param batch_size: The batch size
        :return: None. Updates the weights and biases
        """

        # Once we're done accumulating the gradient, we average it out by dividing by the batch size
        self.dW /= batch_size

        # Finally update the weights
        self.weights -= self.dW

        """
        Updating the biases
        """
        self.biases -= learning_rate * error


    def reset_grad(self):
        """
        A function used to reset the gradient accumulator back to 0
        :return: None. Resets the gradient
        """
        self.dW = np.zeros(self.output_size, self.input_size)


    def __repr__(self):
        return f"Layer(in={self.input_size}, out={self.output_size})"


class Network:
    def __init__(
            self,
            input_size,
            output_size,
            hidden_layer_count,
            hidden_layer_size,
            activation_function,
            output_activation,
    ):
        if hidden_layer_count == 0:
            hidden_layer_size = input_size

        self.hidden_layers = ([Layer(input_size, hidden_layer_size, activation_function)] +
                              [Layer(hidden_layer_size, hidden_layer_size, activation_function)
                               for _ in range(hidden_layer_count - 1)])

        self.output_layer = Layer(hidden_layer_size, output_size, output_activation)

        self.network = self.hidden_layers + [self.output_layer]

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_count= hidden_layer_count
        self.hidden_layer_size = hidden_layer_size
        self.activation_function = activation_function
        self.output_activation = output_activation

    def forward(self, inputs):
        """
        Forward pass for the network
        :return: None. It calculates the output of the network with the given inputs
        """
        self.network[0].set_inputs(inputs)
        for i in range(0,len(self.network) - 1):
            # Basically feed forward of the first layer, then put those outputs as in the inputs of the next layer
            self.network[i].forward()
            self.network[i+1].set_inputs(self.network[i].outputs[1])
        self.output_layer.forward()

    def show_output(self):
        """
        Shows the final output of the entire network's calculations
        :return: Prints the output layer's outputs
        """
        print('<Network> Final output : ', self.output_layer.outputs[1])

    def return_output(self):
        """
        Returns the final output of the entire network's calculations
        :return: the output layer's outputs
        """
        return self.output_layer.outputs[1]

    def backpropagation(self, output_error, learning_rate : int):
        """
        A method used for a single backwards pass using backpropagation using a single label
        :param learning_rate: The learning rate
        :param labels: The iterable of training labels
        :param output_error: The error found via calculating the gradient on a batch of data
        :return: Sets the weights closer to the correct value for optimal classification
        """

        """
        This process will take place in two steps:
            1. Doing backpropagation from the outputs
            2. Doing backpropagation between hidden layers 
        """

        """
        Step 1
        First we need to do the backpropagation at the output layer
        """

        # We need to make sure to calculate the backpropagation gradient BEFORE changing
        # the weights because we need to be learning from the weights that made the wrong prediction
        backprop_gradient = self.output_layer.weights.T @ output_error

        """
        Updating the output weights
        """
        dW_output = learning_rate * np.outer(output_error, self.output_layer.inputs)
        self.output_layer.weights -= dW_output

        """
        Updating the biases
        """
        self.output_layer.biases -= learning_rate * output_error

        """
        Step 2
        Then we need to propagate the error back through the network using the newly obtained error term
        """
        network = self.network[:-1]

        # Go through all the layers (except the output layer), find the error and update the weights accordingly
        # using each layer's backprop method
        for layer in reversed(network):
            backprop_gradient = layer.backward(
                upstream_gradient=backprop_gradient,
                learning_rate=learning_rate
            )

    def mini_batch_grad_desc(self, learning_rate, data, epochs, labels, batch_size):
        """
        A method to run stochastic mini-batch gradient descent on the network by calling the prior
        train method
        :param learning_rate: The learning rate
        :param data: The training data
        :param epochs: The number of epochs
        :param labels: The labels of the data
        :param batch_size: The size of the mini-batch
        :return: None
        """
        import random
        """
        Mini batch-gradient descent steps:
         - Shuffle the datapoints via indices,
         - Split those indices into batches,
         - Find the gradient based on all of those datapoints in each batch
         - Sum up and average the gradient
         - Do a single weight update and repeat for every batch (1 epoch) 
        """

        # Find the gradient based on all of those datapoints in each batch
        # - outer loop : epochs
        # - inner loop : batches
        # - inner inner loop : sum up gradients in each batch

        # Then differentiating the activation function
        def d_output_active(x):
            return self.output_layer.activation(x, True)

        # Iterates epochs number of times
        for epoch in range(epochs):
            # Shuffle the datapoints via indices
            r_indices = random.sample(range(len(data)), len(data))
            output_error = np.zeros(self.output_size)
            # Based on the batch size, will iterate through all the data using len(data)/batch_size iterations
            for batch in range(0, len(data), batch_size):
                # Sum up and calculate the gradient
                dL = 0
                for dp in range(batch,batch+batch_size):
                    # We need the network's current predictions with a forward pass
                    self.forward(data[r_indices[dp]])

                    # After differentiating the cost function, we have an expression that we use to find the
                    # error across all datapoints in the batch and then averaging them to find the final error
                    dL += (self.output_layer.outputs[1] - labels[r_indices[dp]])

                    # The error vector to be propagated through the network is computed as such
                    # which should be the size of the output layer
                    output_error = dL * d_output_active(self.output_layer.outputs[0])


                # Done once a batch
                self.backpropagation(output_error=output_error, learning_rate=learning_rate)

    def singleton_grad_desc(self, learning_rate, data, epochs, labels):
        self.mini_batch_grad_desc(learning_rate=learning_rate, data=data, epochs=epochs, labels=labels, batch_size=1)



    def __repr__(self):
        return f" input ({self.input_size}) | hidden layers ({self.hidden_layer_count}) ({self.hidden_layer_size}) | output layer ({self.output_size})"
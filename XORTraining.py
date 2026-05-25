import pickle

import numpy as np

from ExternalFunctions import tanh
from Network import Network

x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
])

xor_network = Network(
    input_size=2,
    output_size=1,
    hidden_layer_size=3,
    hidden_layer_count=1,
    activation_function=tanh,
    output_activation=tanh,
)

xor_network.mini_batch_grad_desc(
    learning_rate=0.2,
    data = x,
    labels=y,
    epochs=2000,
    batch_size=4,
)

pickle.dump(xor_network, open('xor_network.p', 'wb'))
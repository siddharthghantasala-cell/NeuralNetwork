import numpy as np

def sigmoid(x, derivative : bool = False):
    if derivative:
        return sigmoid(x, False) * (1 - sigmoid(x, False))
    return 1.0/(1.0 + np.exp(-x))

def relu(x, derivative : bool = False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.where(x > 0, x, 0)

def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)
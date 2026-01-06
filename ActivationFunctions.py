import numpy as np


def stepFunc(x):
    return np.where(x < 0, 0, 1)


def sigmoid(x, derivative : bool = False):
    if derivative:
        return sigmoid(x, True) * (1 - sigmoid(x, True))
    return 1.0/(1.0 + np.exp(-x))

def relu(x, derivative : bool = False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.where(x > 0, x, 0)

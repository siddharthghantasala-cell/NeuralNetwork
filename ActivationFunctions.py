import numpy as np


def stepFunc(x):
    return np.where(x < 0, 0, 1)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(x):
    return x if x > 0 else 0
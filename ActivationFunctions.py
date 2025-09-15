import numpy as np


def stepFunc(input : int):
    if input < 0:
        return 0
    else:
        return 1


def sigmoid(input : int):
    return 1.0/(1.0 + np.exp(-input))

def relu(input : int):
    return input if input > 0 else 0
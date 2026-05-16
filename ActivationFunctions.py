import numpy as np

def sigmoid(x : np.ndarray, derivative : bool = False) -> int:
    if derivative:
        return sigmoid(x, False) * (1 - sigmoid(x, False))
    return 1.0/(1.0 + np.exp(-x))

def relu(x : np.ndarray, derivative : bool = False) -> int:
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.where(x > 0, x, 0)

def softmax(x : np.ndarray, temperature=1.0) -> int:
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)
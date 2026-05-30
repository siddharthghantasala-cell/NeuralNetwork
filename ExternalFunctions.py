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

def tanh(x : np.ndarray, derivative : bool = False) -> int:
    if derivative:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)

def cross_entropy(predicted : np.ndarray, actual : np.ndarray, is_softmax : bool, derivative : bool = False) -> float:
    if is_softmax:
        if derivative:
            return -(actual - predicted)
        else:
            return -(actual @ np.log(predicted.T + 1e-9))
    else:
        if derivative:
            return -(actual / predicted)
        else:
            return -(actual @ np.log(predicted.T + 1e-9))

# Initializations
def he_initialization(input_size, output_size):
    return np.sqrt(2.0/input_size)

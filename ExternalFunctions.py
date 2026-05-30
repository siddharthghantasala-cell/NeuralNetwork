import numpy as np

# Activation Functions
def sigmoid(x : np.ndarray, upstream_gradient, derivative : bool = False) -> int:
    if derivative:
        return sigmoid(x, upstream_gradient,False) * (1 - sigmoid(x, upstream_gradient, False))
    return upstream_gradient * 1.0/(1.0 + np.exp(-x))

def relu(x : np.ndarray, upstream_gradient, derivative : bool = False) -> int:
    if derivative:
        return upstream_gradient * np.where(x > 0, 1, 0)
    return upstream_gradient * np.where(x > 0, x, 0)

def softmax(x : np.ndarray, derivative : bool, upstream_gradient, temperature=1.0) -> int:
    e_x = np.exp((x - np.max(x, axis=0, keepdims=True)) / temperature)
    s = e_x / e_x.sum(axis=0)
    if derivative:
        return s * (upstream_gradient - np.sum(s * upstream_gradient, axis=0, keepdims=True))
    return s

def tanh(x : np.ndarray, upstream_gradient, derivative : bool = False) -> int:
    if derivative:
        return upstream_gradient * (1 - np.tanh(x) ** 2)
    return upstream_gradient * np.tanh(x)

# Loss functions
def mse(predicted, actual, derivative : bool = True) -> float:
    if derivative:
        return actual - predicted
    else:
        return np.round(((predicted - actual) ** 2).mean())

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

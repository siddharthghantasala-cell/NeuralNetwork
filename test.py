import numpy as np
from ActivationFunctions import *
arr = np.array([-1, 2, -3, 4, 5, -6])
modified_step = np.vectorize(stepFunc)
print(modified_step(arr))
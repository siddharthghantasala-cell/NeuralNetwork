import numpy as np

a = np.array(
    [
        [2,2,2],
        [2,2,2]
     ]
)

print(a)

a[:,0:2] *= 2
print(a.shape)

"""
in numpy array indexing, it goes like
array[row_starting : row_ending, column_starting : column_ending]
"""
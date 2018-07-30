import numpy as np

def one_hot_encode(input):
    n = len(input)
    k = max(input) + 1
    out = np.zeros(n,k)

    for item in range(n):
        out[item, input[n]] = 1

    return out 

one_hot_encode(np.array([5,3,2,1]))

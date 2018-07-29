import numpy as np
import matplotlib.pyplot as plt

num = 4
dim = 2

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ])

targs = np.array([0,1,1,0])

ones = np.array([[1]*num]).T

plt.scatter( X[:, 0], X[:,1], c=targs)
plt.show()

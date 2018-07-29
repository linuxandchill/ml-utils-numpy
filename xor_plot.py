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

#turn it into 3D problem in order to solve it
#can draw a plane between datasets w 3D

xy = np.matrix(X[:,0] * X[:, 1]).T
Xb = np.array(np.concatenate((ones, xy, X), axis=1))

plt.scatter( X[:, 0], X[:,1], c=targs)
plt.show()

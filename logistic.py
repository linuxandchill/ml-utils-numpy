import numpy as np

rows = 10
cols = 2

X = np.random.randn(rows,cols)

biases = np.ones((rows, 1), dtype=int)

Xb = np.concatenate((biases, X), axis=1)

W = np.random.randn(cols + 1)

z = np.dot(Xb, W)


def sigmoid(z):
    return 1/(1 + np.exp(-z))

out = sigmoid(z)

print(out)

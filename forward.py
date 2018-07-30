import numpy as np
import matplotlib.pyplot as plt

#samples_per_class
spc = 1000

#clouds

X1 = np.random.randn(spc,2 ) + np.array([0,-2])
X2 = np.random.randn(spc,2 ) + np.array([2, 2])
X3 = np.random.randn(spc,2 ) + np.array([-2,2])
X = np.vstack([X1, X2, X3])
#print(X.stack) => (3000,2)


labels = np.array([0]*spc + [1]*spc + [2]*spc)

#plt.scatter(X[:,0], X[:,1], c=labels, s=50, alpha=0.4)
#plt.show()

D = 2
M = 3
K = 3
# randomly initialize weights & biases 
weight1 = np.random.randn(D, M)
bias1 = np.random.randn(M)
weight2 = np.random.randn(M, K)
bias2 = np.random.randn(K)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward_pass(X, weight1, bias1, weight2, bias2):
    #sigmoid
    z = sigmoid(np.dot(X, weight1) + bias1)
    A = (np.dot(z, weight2)) + bias2
    expA = np.exp(A)

    Y = expA / expA.sum(axis = 1, keepdims=True)
    return Y


#print(forward_pass(X, weight1, bias1, weight2, bias2))






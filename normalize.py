import numpy as np

#samples_per_class
spc = 1000

#clouds

X1 = np.random.randn(spc,2 ) + np.array([0,-2])
X2 = np.random.randn(spc,2 ) + np.array([2, 2])

X[:, 0] = (X[:, 0] - X[:,0].mean()) / X[:,0].std()
X[:, 1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

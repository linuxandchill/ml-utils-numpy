import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(100,2)

X[:50, : ] = X[:50, :] - 8*np.ones((50,2))
X[50:, : ] = X[50:, :] + 8*np.ones((50,2))

T = np.array([0]*50 + [1]*50)

plt.scatter(X[:,0], X[:,1] , c=T, s=100, alpha=0.5)

#draw separator line
x_ = np.linspace(-5,5,100) 
y_ = -x_

plt.plot(x_, y_)
plt.show()


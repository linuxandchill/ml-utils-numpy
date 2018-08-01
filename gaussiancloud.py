import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(100,2)

x[:50, :] = x[:50, :] - 8*np.ones((50,2))
x[50:, :] = x[50:, :] + 8*np.ones((50,2))

t = np.array([0]*50 + [1]*50)

plt.scatter(x[:,0], x[:,1] , c=t, s=100, alpha=0.5)

#draw separator line
x_ = np.linspace(-5,5,100) 
y_ = -x_

plt.plot(x_, y_)
plt.show()


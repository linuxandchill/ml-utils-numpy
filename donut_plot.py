import numpy as np
import matplotlib.pyplot as plt

num = 1000
dim = 2

inner_radius = 5
outer_radius = 10

r1 = np.random.randn(num//2) + inner_radius
theta = 2*np.pi * np.random.random(num//2)
X_inner = np.concatenate([[r1 * np.cos(theta)], [r1 * np.sin(theta)]]).T

r2 = np.random.randn(num//2) + outer_radius
theta = 2 * np.pi * np.random.random(num//2)
X_outer = np.concatenate([[r2 * np.cos(theta)], [r2 * np.sin(theta)]]).T

X = np.concatenate([ X_inner, X_outer ])
T = np.array([0] * (num//2) + [1]*(num//2))

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()


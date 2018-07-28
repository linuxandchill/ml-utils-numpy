import numpy as np

'''
rows = 20
cols = 2

X = np.random.randn(rows, cols)

X[:20, :] = X[:20, :] - 2*np.ones((20,cols)) # centering x & y -2
X[20:, :] = X[20:, :] + 2*np.ones((20,cols)) # centering x & y around +2

target = np.array([[1]*rows])

ones = np.array([[1]*rows]).T
Xb = np.concatenate((ones, X), axis = 1)
'''

def crossEntropyLoss(target, y_pred):
    err = 0 
    for i in range(rows): #sum over each err for each sample
        if target[i] == 1:
            err -= np.log(y_pred[i])
        else: ## if targ = 0
            err -= np.log(1 - y_pred[i])

    return err


    

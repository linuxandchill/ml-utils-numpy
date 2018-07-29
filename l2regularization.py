import numpy as np


### GOAL::: PREVENT WEIGHTS FROM BLOWING UP AND GOING TO -INF, +INF

### during training loop

#calculate loss
## loss_fn(targ, y)

lambda_const = 0.1
lr = 0.1

w += lr * (np.dot((targ - Y).T, Xb) - lambda_const*w)
#call sigmoid(Xb.dot(w))



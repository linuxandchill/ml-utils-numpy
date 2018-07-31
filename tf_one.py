import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#samples_per_class
spc = 1000

#clouds

X1 = np.random.randn(spc,2 ) + np.array([0,-2])
X2 = np.random.randn(spc,2 ) + np.array([2, 2])
X3 = np.random.randn(spc,2 ) + np.array([-2,2])
X = np.vstack([X1, X2, X3])
#print(X.stack) => (3000,2)

Y = np.array([0]*spc + [1]*spc + [2]*spc)

#plt.scatter(X[:,0], X[:,1], c=labels, s=50, alpha=0.4)
#plt.show()

D = 2
M = 3
K = 3

N = len(Y)

targs = np.zeros((N, K))

for i in range(N):
    targs[i, Y[i]] = 1

#initialize weights
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward_pass(X, W1, b1, W2, b2):
    z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(z, W2) + b2

#placeholder vars
tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M]) # create symbolic variables
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

logits = forward_pass(tfX, W1, b1, W2, b2)

#calculates gradients and does GD automatically
#no need to specify derivative
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tfY, logits=logits))

#Train Function
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(logits, axis=1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000): 
    #training
    sess.run(train_op, feed_dict={tfX: X, tfY: targs})
    #predict
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: targs})
    if i % 10 == 0:
        print(np.mean(Y == pred))


import tensorflow as tf
import numpy as np

N,D = X_train.shape
K = 10

X = tf.placeholder(tf.float32, shape=(None, D), name="x")
Y = tf.placeholder(tf.float32, shape=(None, K), name="y")

weight_1 = tf.Variable(tf.random_normal(shape=(D,200))
bias_1 = np.zeros(200)

#use reduce_sum to reduct cost func
cost = tf.reduce_sum

train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)


import tensorflow as tf
import numpy as np

#placeholders for input X & Y
#variables for biases and weights

X = tf.placeholder(tf.float32, shape=(None, D), name='x')
Y = tf.placeholder(tf.float32, shape=(None, K), name='y')
W1 = tf.Variable(W1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32))
W2 = tf.Variable(W2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))
W3 = tf.Variable(W3_init.astype(np.float32))
b3 = tf.Variable(b3_init.astype(np.float32))

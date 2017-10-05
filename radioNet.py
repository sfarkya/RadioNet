import tensorflow as tf
import numpy as np

def radioNet(x,keep_prob):
## defining model ###
    ## layer 1
    W_conv1 = weight_variable([1,3,1,64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
    d_conv1 = dropout(h_conv1,keep_prob)

    #print d_conv1
    ## layer 2
    W_conv2 = weight_variable([2,3,64,16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2)+b_conv2)
    d_conv2 = dropout(h_conv2,keep_prob)
    flattened = tf.reshape(d_conv2,[-1,16*2*128])

    ## dense layer
    W_fc1 = weight_variable([16*2*128,128])
    b_fc1 = bias_variable([128])
    h_fc1 = tf.nn.relu(tf.matmul(flattened,W_fc1)+b_fc1)
    d_fc1 = dropout(h_fc1,keep_prob)

    ## final connected layer
    W_fc2 = weight_variable([128,10])
    b_fc2 = bias_variable([10])
    y = tf.matmul(h_fc1,W_fc2) + b_fc2

    return y


## This area is dedicated to for the functions Required in the code
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)

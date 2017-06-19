import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def gen_batch(size):
    kernel_size = 3
    image_size = 64
    kernel = np.ones([kernel_size,kernel_size])
    x = np.random.randint(0, image_size-kernel_size+1)
    y = np.random.randint(0, image_size-kernel_size+1)
    result = np.zeros([image_size, image_size])
    result[x:x+kernel_size, y:y+kernel_size] = kernel
    return [result, [x,y]]

temp, _ = gen_batch(1)
image_size = len(temp)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_tensor = tf.reshape(x, [-1,image_size,image_size,1])
h_conv1 = tf.nn.tanh(conv2d(x_tensor, W_conv1) + b_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.tanh(conv2d(h_conv1, W_conv2) + b_conv2)

W_fc1 = weight_variable([image_size*image_size*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_conv2, [-1, 7*7*64])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def gen_batch(size):
    n = 4
    b = 0.1
    x = 2*np.pi*n*np.random.rand(size,1)
    y = np.cos(x) + b*np.random.rand(size,1) + 0*np.ones([size,1])
    return [x,y]

sess = tf.InteractiveSession()
topology = [1, 20, 20, 200, 200, 200, 200, 20, 20, 20, 20, 1]

x = tf.placeholder(tf.float32, shape=[None, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])


for i in range(len(topology)-1):
    temp = tf.random_uniform([topology[i], topology[i+1]], -1, 1)
    weight = tf.Variable(temp)
    temp = tf.random_uniform([topology[i+1]], -1, 1)
    bias = tf.Variable(temp)
    if i==0:
        temp = x
    else:
        temp = output
    output = tf.nn.tanh(tf.matmul(temp, weight) + bias)

loss = tf.reduce_mean(tf.squared_difference(output, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

for i in range(2000):
  batch = gen_batch(50)
  if i%100 == 0:
    train_loss = loss.eval(feed_dict={x:batch[0], y_: batch[1]})
    print("step %d, training loss %g"%(i, train_loss))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

for i in xrange(10):
    testSet = gen_batch(500)
    print("test loss %g"%loss.eval(feed_dict={x: testSet[0], y_: testSet[1]}))


n = 4
size = 1000
x2 = np.transpose([np.linspace(0, 2*np.pi*n, size)])
y2 = np.cos(x2)
y3 = output.eval(feed_dict={x:x2})
plt.plot(x2, y2, x2, y3)
plt.show()

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

points_num = 100
vectors = []

for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors] #真实的点的x坐标
y_data = [v[1] for v in vectors] #真实的点的y坐标

plt.plot(x_data, y_data, 'r*', label="Orignal data")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

for step in range(100):
    sess.run(train)
    print("Step=%d, Loss=%f, [Weight=%f Bias=%f]"%(step, sess.run(loss), sess.run(W), sess.run(b)))

plt.plot(x_data, y_data, 'r*', label="Orignal data")
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted line")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sess.close()
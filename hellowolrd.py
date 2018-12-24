import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



hw = tf.constant('Hello World! I love Tensorflow')
a = tf.constant(2)
b = tf.constant(3)
c = tf.multiply(a, b)
d = tf.add(c, 1)

with tf.Session() as sess:

#运行 Graph
    print(sess.run(d))
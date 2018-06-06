# coding:utf-8
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([[1.0,2.0]])
b = tf.constant([[3.0],[4.0]])

y = tf.matmul(a,b)
print (y)

with tf.Session() as sess:
	print (sess.run(y))
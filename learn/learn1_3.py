#coding:utf-8
#两层简单神经网络（全连接）
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#定义输入和参数
x = tf.placeholder(tf.float32, shape=(1, 2))# 先占位，后在sess.run中喂食一组数据，2为一组数据有两个变量
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))# 前节点为2，后节点为3
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))# 前节点为3，后节点为1

#定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#用会话计算
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()# 初始化所有节点（简写）
	sess.run(init_op)
	print ("y in learn1_3.py is:\n",sess.run(y, feed_dict = {x: [[0.7, 0.5]]}))

'''
y in learn1_2.py is:
 [[3.0904665]]
 '''
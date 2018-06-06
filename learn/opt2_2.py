#coding:utf-8
import tensorflow as tf
import numpy as np# python的科学计算模块
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_SIZE = 8
seed = 23455
COST = 1
PROFIT = 9
#成本1元，利润9元，则预测越少损失越大，结果会往多了进行预测

#基于seed产生的随机数
rdm = np.random.RandomState(seed)
X = rdm.rand(32, 2)
#作为输入数据集的标签（正确答案）
Y = [[x0 + x1 + (rdm.rand()/10.0 - 0.05)] for (x0, x1) in X]
print("X:\n", X)
print("Y:\n", Y)

#1定义神经网络的输入、参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1= tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))

y = tf.matmul(x, w1)

#2定义损失函数及反向传播方法
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST*(y-y_),PROFIT*(y_-y)))
learning_rate = 0.001
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#monentum = 0.9
#train_step = tf.train.MomentumOptimizer(learning_rate,monentum).minimize(loss)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#三个不同的优化器

#3生成会话，训练STEPS轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()# 初始化所有节点（简写）
	sess.run(init_op)
	STEPS = 20000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict = {x: X[start:end], y_: Y[start:end]})
		if i % 500 == 0:
			print("After %d training step(s), w1 is:" % (i))
			print(sess.run(w1))

	print("\n")
	print("final w1 is:\n", sess.run(w1))

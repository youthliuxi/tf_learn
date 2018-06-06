#coding:utf-8
import tensorflow as tf
import numpy as np# python的科学计算模块
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
BATCH_SIZE = 8
seed = 23455

#基于seed产生的随机数
rng = np.random.RandomState(seed)
#随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入训练集
X = rng.rand(32, 2)
#从X这个32行2列的矩阵中取出一行 判断如果和小于1 给Y赋值为1 如果和不小于1 给y赋值0
#作为输入数据集的标签（正确答案）
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]# 实例教学，假设标准是：当x0+x1的值小于1时未合格产品，其他为不合格产品
print("X:\n", X)
print("Y:\n", Y)

#1定义神经网络的输入、参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))
y_= tf.placeholder(tf.float32, shape=(None, 1))

w1= tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2= tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#2定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y-y_))
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
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))
	print("\n")

	STEPS = 30000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict = {x: X[start:end], y_: Y[start:end]})
		if i % 500 == 0:
			total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
			print("After %d training step(s), loss on all data is %g" % (i, total_loss))

	print("\n")
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))

'''
X:
 [[0.83494319 0.11482951]
 [0.66899751 0.46594987]
 [0.60181666 0.58838408]
 [0.31836656 0.20502072]
 [0.87043944 0.02679395]
 [0.41539811 0.43938369]
 [0.68635684 0.24833404]
 [0.97315228 0.68541849]
 [0.03081617 0.89479913]
 [0.24665715 0.28584862]
 [0.31375667 0.47718349]
 [0.56689254 0.77079148]
 [0.7321604  0.35828963]
 [0.15724842 0.94294584]
 [0.34933722 0.84634483]
 [0.50304053 0.81299619]
 [0.23869886 0.9895604 ]
 [0.4636501  0.32531094]
 [0.36510487 0.97365522]
 [0.73350238 0.83833013]
 [0.61810158 0.12580353]
 [0.59274817 0.18779828]
 [0.87150299 0.34679501]
 [0.25883219 0.50002932]
 [0.75690948 0.83429824]
 [0.29316649 0.05646578]
 [0.10409134 0.88235166]
 [0.06727785 0.57784761]
 [0.38492705 0.48384792]
 [0.69234428 0.19687348]
 [0.42783492 0.73416985]
 [0.09696069 0.04883936]]
Y:
 [[1], [0], [0], [1], [1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [0], [0], [1], [1], [0], [1], [0], [1], [1], [1], [1], [1], [0], [1]]
w1:
 [[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
w2:
 [[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]


After 0 training step(s), loss on all data is 5.13118
After 500 training step(s), loss on all data is 0.429111
After 1000 training step(s), loss on all data is 0.409789
After 1500 training step(s), loss on all data is 0.399923
After 2000 training step(s), loss on all data is 0.394146
After 2500 training step(s), loss on all data is 0.390597


w1:
 [[-0.7000663   0.91363174  0.0895357 ]
 [-2.3402493  -0.14641264  0.58823055]]
w2:
 [[-0.06024268]
 [ 0.91956186]
 [-0.06820709]]
 '''
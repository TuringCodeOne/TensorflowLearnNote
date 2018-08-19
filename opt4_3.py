#coding:utf-8
#酸奶成本9元，酸奶利润1元
#预测多了损失大

#0.导入模块，生成数据集
import tensorflow as tf
import numpy as np
batchSize = 8
SEED = 23445
COST = 9
PROFIT = 1

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10-0.05)] for (x1,x2) in X]

#1,定义神经网络的输入、参数和输出，定义前向传播的过程
x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed = 1))
y = tf.matmul(x,w1)

#2,定义损失函数以及反向传播方法
#重新定义损失函数，使得预测多了的损失大，于是模型应该偏向少的方向预测
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
trainStep = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

#3.执行会话，训练模型
with tf.Session() as sess:
	initOp = tf.global_variables_initializer()
	sess.run(initOp)
	STEPS = 100000
	for i in range(STEPS):
		start = (i*batchSize)%32
		end = (i*batchSize)%32 + batchSize
		sess.run(trainStep,feed_dict={x: X[start:end],y_: Y_[start:end]})
		if i % 500 == 0:
			print "After %d training steps, w1 is:" %(i)
			print sess.run(w1),"\n"
	print "Final w1 is:",sess.run(w1)

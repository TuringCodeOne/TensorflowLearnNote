#coding:utf-8
#设损失函数 loss=(w+1)^2, 令w初值是常熟5.反向传播就是求最优w,即求最小loss对应的w值
import tensorflow as tf

#定义待优化的参数w初值为5
w = tf.Variable(tf.constant(5,dtype=tf.float32))

#定义损失函数
loss = tf.square(w+1)

#定义反向传播方法
trainStep = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#生成会话，训练40轮
with tf.Session() as sess:
	initOp = tf.global_variables_initializer()
	sess.run(initOp)
	for i in range(35):
		sess.run(trainStep)
		valW = sess.run(w)
		valLoss = sess.run(loss)
		print "After %d steps: w is %f, loss is %f" %(i,valW,valLoss)

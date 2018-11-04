# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-29 21:30:16
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-03 20:23:50
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
def add_layer(inputs,in_size,out_size,activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs,Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

# data
x_data = np.linspace(-1,1,300, dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.04,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# placeholder 理解为输入
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#网络
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

#loss 和 优化器
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show() 

with tf.Session() as sess:
	sess.run(init)
	for i in range(1000):
		sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
		if i%50 == 0:
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			prediction_value = sess.run(prediction,feed_dict={xs:x_data})
			lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
			plt.pause(1)
			print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
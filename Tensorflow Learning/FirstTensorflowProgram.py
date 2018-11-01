# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-29 10:36:42
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-10-29 10:47:42


import tensorflow as tf 
import numpy as np 

x_data = np.random.rand(100).astype(np.float32) 
y_data = x_data * 0.1 + 0.3

# create model

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data)) 
optimizer = tf.train.GradientDescentOptimizer(0.5) 	
train = optimizer.minimize(loss)

#initialize all variable
init = tf.global_variables_initializer() 
#create Session
sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step%20 == 0:
		print(step,sess.run(Weights),sess.run(biases))
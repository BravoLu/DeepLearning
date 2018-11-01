# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-29 21:30:16
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-10-29 21:35:04

import tensorflow as tf 
def add_layer(inputs,in_size,out_size,activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs,Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs
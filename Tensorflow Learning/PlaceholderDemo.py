# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-29 21:24:21
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-10-29 21:27:32

import tensorflow as tf 
input1 = tf.placeholder(tf.float32) 
input2 = tf.placeholder(tf.float32) 

output = tf.multiply(input1,input2) 


# 需要传入的值放在feed_dict={}中
with tf.Session() as sess: 
	print(sess.run(output, feed_dict={input1:[7,2],input2:[8,1]}))
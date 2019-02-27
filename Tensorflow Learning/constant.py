# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-01-17 10:19:20
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-01-17 10:21:21
import tensorflow as tf 

a = tf.constant([0.0,0.0,1.0,1.0], dtype=tf.float32, shape=[1,1,4])
print(a)

with tf.Session() as sess:
	print(sess.run(a))
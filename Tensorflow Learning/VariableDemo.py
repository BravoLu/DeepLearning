# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-29 21:18:39
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-10-31 20:35:17

import tensorflow as tf 
state = tf.Variable(0, name='counter')

one = tf.constant(1)
new_value = tf.add(state,one)

update = tf.assign(state, new_value)

# 如果定义Variable,就一定要Initialize
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		#直接print(state)不起作用，一定要把sess的指针指向state再进行print
		print(sess.run(state))

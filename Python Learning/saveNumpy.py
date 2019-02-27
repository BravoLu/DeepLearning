# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-12-07 23:20:06
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-12-09 19:06:01
import numpy as np

np_list = []

for i in range(10):

	a = np.random.rand(1,10)
	np.save('test.npy',a)

	c = np.load('test.npy')
	print('round{}\n'.format(i))
	print(c)



#  save data
[[face_ID, feature] ,..., ]
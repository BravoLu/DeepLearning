# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-01 14:54:12
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-01 14:56:14

import pdb

def add(a,b):
	return a+b

def cal(a,b):
	pdb.set_trace()
	c = add(a,b)
	print(c)


if __name__ == '__main__':
	cal(3,4)
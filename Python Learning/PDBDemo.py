# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-01 14:54:12
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-07 20:48:44

import pdb

# def add(a,b):
# 	return a+b

# def cal(a,b):
# 	pdb.set_trace()
# 	c = add(a,b)
# 	print(c)


# if __name__ == '__main__':
# 	cal(3,4)
def pdb_test(arg):
	for i in range(arg):
		print(i)
	return arg

def test(arg):
	print('hello')
pdb.run("pdb_test(3)")

#在交互环境通常使用Pdb.run来调试

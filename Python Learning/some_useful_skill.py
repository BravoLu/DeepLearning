# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-12-14 11:26:22
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-02-25 14:57:27

# 利用dict代替if-else
def f1(num):
	result = cmp(num,1)
	return {-1:'less',0:'equal',1:'more'}.get(result)


#当定义一个类的时候，提供__repr__可以返回用来表示该类对象的可打印字符串
def __repr__(self):
	return "<some description here>"
#重载小于运算符
def __lt__(self, other):
	return self.__value < other.__value

import sh
sh.pwd() 
sh.mkdir('new_folder')


#collections
from collections import namedtuple, deque, defaultdict, OrderedDict
Point = namedtuple('Point', ['x', 'y'])
p = Point(1,2)

q = deque(['a', 'b', 'c'])
q.append('x')
q.appendleft('y')
q.pop()
q.popleft()

#使用dict时，如果引用的Key不存在，就会抛出KeyError。如果希望Key不存在时，返回一个默认值，就可以用defaultdict
dd = defaultdict(lambda:'N/A')
dd['key1'] = 'abc'

#使用dict时，Key是无序的。在对dict做迭代时，我们无法确定key的顺序。
od = OrderedDict([('a',1),('b',2),('c',3)])

Counter
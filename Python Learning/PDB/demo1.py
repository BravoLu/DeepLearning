# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-19 19:13:50
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-19 19:41:04

import pdb

a = "aaa"
b = 'bbb'
a,b = b,a
print(a)
print(b)
#pdb.set_trace()
b = 'bbb'
c = 'ccc'
final = a + b + c
print(final)
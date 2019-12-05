# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-12-04 14:34:27
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-12-04 14:49:39

a = [1,2,3,4,5]
a = map(lambda x:x**2, a)
a = [i for i in a]
print(a)


dict = {'3': 'x', '2':'y'}
dict = sorted(dict.items(), key=lambda k:k[0], reverse=False)
print(dict)
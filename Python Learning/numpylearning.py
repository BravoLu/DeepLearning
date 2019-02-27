# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-12-26 10:33:36
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-12-26 10:53:27

import numpy as np 
indices = np.random.permutation(10)

np.random.shuffle(1)
print(indices)

#对角
np.diag([1,2,3])
#左右翻转 
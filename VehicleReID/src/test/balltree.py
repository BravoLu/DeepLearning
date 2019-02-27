# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-20 20:08:07
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-20 20:17:07

import numpy as np 
from sklearn.neighbors import BallTree
np.random.seed(1)
x = np.random.random((10,3))
a = [1, 2, 3, 4]

print(x)
print(x[:1])
tree = BallTree(x, leaf_size=3)
dist,ind = tree.query(x[:1],k=3)
print(ind)
print(dist)
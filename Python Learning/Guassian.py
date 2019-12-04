# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-05-21 16:25:22
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-05-21 16:30:48
import matplotlib.pyplot as plt
import numpy as np 

x = np.arange(-1,1,0.01) 
sigma = 1.0
u     = 0.1
y = 1. / sigma * np.exp(-1*(x-u)*(x-u)/(2*sigma*sigma))


plt.plot(x,y,'-o')
plt.show()


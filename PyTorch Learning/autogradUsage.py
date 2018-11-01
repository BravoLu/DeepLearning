# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-01 19:34:55
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-01 20:21:24
import torch 
from torch.autograd import Variable 

x = Variable(torch.ones(2,2),requires_grad=True)
y = 2*x+2
print(y.grad)
z = y.mean()
k = y.mean()+1
torch.autograd.backward([z,k])
print(y.grad)
print(x.grad)
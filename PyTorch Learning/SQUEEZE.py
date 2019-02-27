# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-04 11:05:29
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-13 22:19:35
import torch
from torch.nn import functional as F
from torch.autograd import Variable

#
# a = torch.ones(2,16,2,2)
# b = F.avg_pool3d(a,(16,1,1))
# print(a)
# print(b)

input = Variable(torch.Tensor([
								  [[1,2,3,4,5,6,7],[1,1,1,1,1,1,1]],
	                              [[1,1,3,3,4,4,5],[1,1,1,1,1,1,1]],
	                              [[2,2,2,2,3,3,3],[1,1,1,1,1,1,1]],
	                              ]))
print(input)
b = F.avg_pool2d(input, kernel_size=(2,2), 

print(b)
# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2019-04-04 14:02:47
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2019-04-04 14:10:59
import torch

def Print(name=None, value=None):
	print("{}:{}".format(name, value))

N, C, H, W = 1, 3, 2, 4
m1 = torch.ones((N,C,H,W))
Print("m1", m1)

m2 = torch.matmul(m1, m1.permute(0,1,3,2))
Print("m2",m2)

#bmm batch Matrix multiply
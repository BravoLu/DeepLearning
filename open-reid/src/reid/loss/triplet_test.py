# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-28 11:27:58
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-10-28 20:13:36

import torch 
step = 0

def checkpoint(dist):
	global step 
	step += 1
	print("step{}:{}".format(step,dist))

inputs = torch.FloatTensor([[1,2],[3,2]])
n = inputs.size(0)
dist = torch.pow(inputs,2).sum(dim=1,keepdim=True).expand(n,n)
checkpoint(dist)
dist = dist + dist.t()
checkpoint(dist)
dist.addmm_(1,-2,inputs,inputs.t())
checkpoint(dist)
dist = dist.clamp(min=1e-12).sqrt()
checkpoint(dist)


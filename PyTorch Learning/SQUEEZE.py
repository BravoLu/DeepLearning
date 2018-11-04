# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-04 11:05:29
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-04 11:24:42
import torch
a = torch.Tensor(2,3,4)
print(a.shape)
print(a.unsqueeze(1))
print(a.unsqueeze(1).shape)


b = torch.Tensor([1,2,3])
print(b.unsqueeze(1))
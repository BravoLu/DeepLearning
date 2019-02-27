import torch
from torch.autograd import Variable
import math
a = torch.FloatTensor([[[[1],[2],[3],[5]],[[1],[1],[1],[1]]], [[[1],[2],[3],[5]],[[1],[1],[1],[1]]]])
b = torch.FloatTensor([[1,2,3,3],[1,1,2,2]])
#print(a.size())
a = Variable(a,requires_grad=True)
print(a)


n,c,h,w = a.size(0),a.size(1),a.size(2),a.size(3)
b = torch.pow(a,2)
print("sum:")
print(torch.sum(b,dim=2))
b = torch.sum(b,dim=2).unsqueeze(2)
print("before:")
print(b)
b = b.expand_as(a)
print("after:")
print(b)
d = b+1
d = 1.0 / d.sqrt()
print(d)
print(a)

a = a.mul(d)
print(a)

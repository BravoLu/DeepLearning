import torch
x = torch.Tensor([1,3,4,1])
print(torch.topk(x,1))

#topk = (1,3,2,4,5)

#maxk = max(topk)

#print(maxk)

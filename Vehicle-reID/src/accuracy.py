from __future__ import absolute_import
from utils import to_torch,to_numpy

def accuracy(output, target, topk=(1,)):
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)
    """
        topk(k, dim=None, largest=True, sorted=True)->(Tensor,LongTensor)
        return  value,index
        
        torch.eq(input, other, out=None) -> Tensor
        * input(Tensor) - the tensor to compare
        * other(Tensor or float) - the tensor or value to compare
        * out(Tensor,optional) - the output tensor.Must be a ByteTensor
        Returns : A torch.ByteTensor containing a 1 at each location where comparison is true
        
        torch.sum(input,dim,out=None) -> Tensor
        * input(Tensor) 
        * dim - 
        * out(Tensor, optional)
        
        torch.view(1,-1)  -> Tensor
        *  
    """
    _, pred = output.topk(maxk, 1, True, True)
    
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    
    
    acc = correct[0].float().sum(dim=0)
    acc = to_numpy(acc)[0] * (1. /batch_size)
    
    # --- debug ---
    #print("type {}".format(type(acc)))
    #print("acc {}".format(acc))
    #print("len {}".format(len(acc)))
    #print("sum {}".format(acc))

    # -------------
    #ret = [] 
    #for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        #ret.append(correct_k.mul_(1. / batch_size))

    return acc

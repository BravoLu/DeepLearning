from __future__ import absolute_import
import torch
from torch import nn
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self,gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        log_p = self.ce(inputs, targets)
        p = torch.exp(-log_p)
        loss = (1.0 - p) ** self.gamma * log_p
        return loss.mean()



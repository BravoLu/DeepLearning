from __future__ import absolute_import 
from .triplet import TripletLoss
from .focal import FocalLoss
import torch.nn as nn
__all__ = [
    'TripletLoss',
    'FocalLoss' ,
]

LOSS_SET = {
    'CrossEntropy': nn.CrossEntropyLoss().cuda(),
    'Triplet':TripletLoss(margin=0.5).cuda(),   
    'Focal':FocalLoss().cuda(),
}

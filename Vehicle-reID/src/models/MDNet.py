import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import copy
import torchvision
from torchvision import transforms as T
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
from tqdm import tqdm
from matplotlib.font_manager import *
from collections import defaultdict
from dataset import NBR_ID,NBR_MODELS,NBR_COLORS


 
#mpl.rcParams['axes.unicode_minus'] = False
#mpl.rcParams['font.sans-serif']    = ['SimHei']


class ArcFC(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                s = 30.0,
                m = 0.5,
                easy_margin=False):
        super(ArcFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print('=> in dim: %d, out dim: %d' % (self.in_features, self.out_features))
    
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(self.out_features ,self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
          
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input,p=2), F.normalize(self.weight, p=2))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
    
        #phi:cos(theta+m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class MDNet(torch.nn.Module):
    def __init__(self,
                 num_id=NBR_ID,
                 num_model=NBR_MODELS,
                 num_color=NBR_COLORS):
        super(MDNet, self).__init__()
    
        self.num_id, self.num_model, self.num_color = num_id, num_model, num_color
    
        # Conv1
        self.conv1_1 = torch.nn.Conv2d(in_channels=3,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv1_2 = torch.nn.ReLU(inplace=True)
        self.conv1_3 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv1_4 = torch.nn.ReLU(inplace=True)
        self.conv1_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)
        
        self.conv1 = torch.nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            self.conv1_3,
            self.conv1_4,
            self.conv1_5
        )
    
        # Conv2
        self.conv2_1 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=128,
                                       kernel_size=(3,3),
                                       stride=(1,1),
                                       padding=(1,1))
        self.conv2_2 = torch.nn.ReLU(inplace=True)
        self.conv2_3 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=128,
                                       kernel_size = (3, 3),
                                       stride = (1,1),
                                       padding=(1,1))
        self.conv2_4 = torch.nn.ReLU(inplace=True)
        self.conv2_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)
        self.conv2 = torch.nn.Sequential(
            self.conv2_1,
            self.conv2_2,
            self.conv2_3,
            self.conv2_4,
            self.conv2_5
        )

    
        # Conv3
        self.conv3_1 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,  
                                       kernel_size=(3,3),
                                       stride=(1,1),    
                                       padding=(1,1))
        self.conv3_4 = torch.nn.ReLU(inplace=True)
        self.conv3_5 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(3,3),
                                       stride=(1,1),
                                       padding=(1,1))
        self.conv3_6 = torch.nn.ReLU(inplace=True)
        self.conv3_7 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)
        self.conv3 = torch.nn.Sequential(
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv3_4,
            self.conv3_5,
            self.conv3_6,
            self.conv3_7
        )

        self.conv4_1_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_1_2 = torch.nn.ReLU(inplace=True)
        self.conv4_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_1_4 = torch.nn.ReLU(inplace=True)
        self.conv4_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size = (3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_1_6 = torch.nn.ReLU(inplace=True)
        self.conv4_1_7 = torch.nn.MaxPool2d(kernel_size=2,  
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)

        self.conv4_1 = torch.nn.Sequential(
            self.conv4_1_1,
            self.conv4_1_2,
            self.conv4_1_3,
            self.conv4_1_4, 
            self.conv4_1_5, 
            self.conv4_1_6, 
            self.conv4_1_7, 
        )
        
        # Conv4-2
        self.conv4_2_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3,3),
                                         stride=(1,1),
                                         padding=(1,1))
        self.conv4_2_2 = torch.nn.ReLU(inplace=True)
        self.conv4_2_3 = torch.nn.Conv2d(in_channels=512,           
                                         out_channels=512,
                                         kernel_size=(3,3), 
                                         stride=(1,1),
                                         padding=(1,1))
        self.conv4_2_4 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3,3),
                                         stride=(1,1),
                                         padding=(1,1))
        self.conv4_2_6 = torch.nn.ReLU(inplace=True)
        self.conv4_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)

        self.conv4_2 = torch.nn.Sequential(
            self.conv4_2_1,
            self.conv4_2_2,
            self.conv4_2_3,
            self.conv4_2_4,
            self.conv4_2_5,
            self.conv4_2_6,
            self.conv4_2_7,
       ) 

       # Conv5_1
        self.conv5_1_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3,3),
                                         stride=(1,1),
                                         padding=(1,1))
        self.conv5_1_2 = torch.nn.ReLU(inplace=True)
        self.conv5_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3,3),
                                         stride=(1,1),
                                         padding=(1,1))
        self.conv5_1_4 = torch.nn.ReLU(inplace=True)
        self.conv5_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,  
                                         kernel_size=(3,3),
                                         stride=(1,1),              
                                         padding=(1,1))
        self.conv5_1_6 = torch.nn.ReLU(inplace=True)
        self.conv5_1_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)
        
        self.conv5_1 = torch.nn.Sequential(
            self.conv5_1_1,
            self.conv5_1_2,
            self.conv5_1_3,
            self.conv5_1_4,
            self.conv5_1_5,
            self.conv5_1_6,
            self.conv5_1_7,
        )
        
        # Conv5_2
        self.conv5_2_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3,3),
                                         stride=(1,1),
                                         padding=(1,1))
        self.conv5_2_2 = torch.nn.ReLU(inplace=True)
        self.conv5_2_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3,3),
                                         stride=(1,1),
                                         padding=(1,1))
        self.conv5_2_4 = torch.nn.ReLU(inplace=True)
        self.conv5_2_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,  
                                         kernel_size=(3,3),
                                         stride=(1,1),              
                                         padding=(1,1))
        self.conv5_2_6 = torch.nn.ReLU(inplace=True)
        self.conv5_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)
        
        self.conv5_2 = torch.nn.Sequential(
            self.conv5_2_1,
            self.conv5_2_2,
            self.conv5_2_3,
            self.conv5_2_4,
            self.conv5_2_5,
            self.conv5_2_6,
            self.conv5_2_7,
        )
        
        # FC6_1
        self.FC6_1_1 = torch.nn.Linear(in_features=25088,       
                                       out_features=4096,
                                       bias=True)
        self.FC6_1_2 = torch.nn.ReLU(inplace=True)
        self.FC6_1_3 = torch.nn.Dropout(p=0.5)
        self.FC6_1_4 = torch.nn.Linear(in_features=4096,
                                       out_features=4096,
                                       bias=True)
        self.FC6_1_5 = torch.nn.ReLU(inplace=True)
        self.FC6_1_6 = torch.nn.Dropout(p=0.5)

        self.FC6_1 = torch.nn.Sequential(
            self.FC6_1_1,
            self.FC6_1_2,
            self.FC6_1_3,
            self.FC6_1_4,
            self.FC6_1_5,
            self.FC6_1_6
        )

        self.FC6_2_1 = copy_deepcopy(self.FC6_1_1)
        self.FC6_2_2 = copy_deepcopy(self.FC6_1_2)
        self.FC6_2_3 = copy_deepcopy(self.FC6_1_3)
        self.FC6_2_4 = copy_deepcopy(self.FC6_1_4)
        self.FC6_2_5 = copy_deepcopy(self.FC6_1_5)
        self.FC6_2_6 = copy_deepcopy(self.FC6_1_6)
    
        self.FC6_2 = torch.nn.Sequential(
            self.FC6_2_1,
            self.FC6_2_2,
            self.FC6_2_3,
            self.FC6_2_4,
            self.FC6_2_5,
            self.FC6_2_6
        )
    
        # FC7_1
        self.FC7_1 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)
        
        self.FC7_2 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)
        # FC8
        self.FC_8 = torch.nn.Linear(in_features=2000,
                                    out_features=1024)

        self.attrib_classifier = torch.nn.Linear(in_features=1000,
                                                 out_features=NBR_MODELS)
        
        # Arc FC layer for branch_2 and branch_3
        self.arc_fc_br2 = ArcFC(in_features=1000,   
                                out_features=NBR_ID,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)
        self.arc_fc_br3 = ArcFC(in_features=1024,
                                out_features=NBR_ID,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)
        
        self.shared_layers = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.branch_1_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_1,
            self.conv5_1,
        )
        
        self.branch_1_fc = torch.nn.Sequential(
            self.FC6_1,
            self.FC7_1,
        )
    
        self.branch_1 = torch.nn.Sequential(
            self.branch_1_feats,
            self.branch_1_fc
        )
    
        self.branch_2_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_2,
            self.conv5_2
        )
    
        self.branch_2_fc = torch.nn.Sequential(
            self.FC6_1,
            self.FC6_2,
        )

        self.branch_2 = torch.nn.Sequential(
            self.branch_2_feats,
            self.branch_2_fc
        )

    def forward(self, 
                x,
                branch,
                label=None):
        
        if branch == 1:
            x = self.branch_1_feats(x)
            x = x.view(x.size(0), -1)
            x = self.branch_1_fc(x)
            x = self.attrib_classifier(x)
            return x

        elif branch == 2:
            if label is None:   
                print('=> label is None.')
                return None
            x = self.branch_2_feats(x)
            x = x.view(x.size(0), -1)
            x = self.branch_2_fc(x)
            x = self.arc_fc_br2.forward(input=x, label=label)
            return x

        elif branch == 3:
            if label is None:
                print('=> label is None.')
                return None
            branch_1 = self.branch_1_feats(x)
            branch_2 = self.branch_2_feats(x)

            branch_1 = branch_1.view(branch_1.size(0), -1)
            branch_2 = branch_2.view(branch_2.size(0), -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)
        
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)
    
            x = self.FC_8(fusion_feats)
            x = self.arc_fc_br3.forward(input=x, label=label)
        
            return x

        elif branch == 4: # test pre-trained weights
            x = self.branch_1_feats(x)
            x = x.view(x.size(0),-1)
            x = self.branch_1_fc(x)
            
            return x

        elif branch == 5:
            branch_1 = self.branch_1_feats(x)
            branch_2 = self.branch_2-feats(x)
            
            branch_1 = branch_1.view(branch_1.size(0), -1)
            branch_2 = branch_2.view(branch_2.size(0), -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)
        
            fusion_feats = torch.cat((branch_1,branch_2), dim=1)
        
            x = self.FC_8(fusion_feats)
            
            return x

        else:
            print('=> invalid branch')
            return None






















































        























































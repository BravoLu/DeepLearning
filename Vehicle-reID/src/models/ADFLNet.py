from __future__ import absolute_import 
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torch
from dataset import NBR_ID
class CBLR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope,
                 kernel_size=3,
                 stride=1):
        super(CBLR, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride 
        self.block = nn.Sequential(
                        nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(negative_slope=negative_slope,inplace=True)
        )
        
    def forward(self,x):
        x = self.block(x) 

        return x

class SDU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 negative_slope):
        super(SDU, self).__init__()
        
        #self.CBLR_1 = CBLR(in_channels,in_channels,negative_slope)
        self.CBLR_1 = CBLR(in_channels,in_channels,negative_slope)
        self.CBLR_2 = CBLR(2*in_channels,in_channels,negative_slope)
        self.CBLR_3 = CBLR(3*in_channels,out_channels,negative_slope)

    def forward(self,x):
        cblr1 = self.CBLR_1(x)
        cat1 = torch.cat((cblr1, x), dim=1)
        cblr2 = self.CBLR_2(cat1)
        cat2 = torch.cat((cblr2,cat1), dim=1)
        cblr3 = self.CBLR_3(cat2)
        
        return cblr3

class SpatialNormalizationLayer(nn.Module):
    def __init__(self):
        super(SpatialNormalizationLayer,self).__init__()

    def forward(self,x):
        #n,c,h,w = x.size(0),x.size(1),x.size(2),x.size(3) 
        norm = torch.pow(x,2)
        norm = torch.sum(norm,dim=2).unsqueeze(2)
        norm = norm.expand_as(norm)
        norm = norm + 1
        norm = 1.0 / norm.sqrt()
        x = x.mul(norm)
        return x

class ADFLNet(nn.Module):
    def __init__(self):
        super(ADFLNet, self).__init__()
        
        self.Conv0 = nn.Sequential(
                        nn.Conv2d(3,64,3,1,1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(negative_slope=0.15,inplace=True)
        )
        
        self.SDU1 = SDU(64,64, 0.15)
        self.SDU2 = SDU(64,128, 0.15)
        self.SDU3 = SDU(128,192, 0.15)
        self.SDU4 = SDU(192,256, 0.15)
        self.SDU5 = SDU(256,320, 0)
        self.SN   = SpatialNormalizationLayer()
        self.classifier = nn.Linear(1280,NBR_ID)
    def forward(self,x):
        x = self.Conv0(x)
        x = self.SDU1(x)
        #input, kernel_size, stride, padding, output_size
        x = F.max_pool2d(x,3,2,1)
        x = self.SDU2(x)
        x = F.max_pool2d(x,3,2,1)
        x = self.SDU3(x)
        x = F.max_pool2d(x,3,2,1)
        x = self.SDU4(x)
        x = F.max_pool2d(x,3,2,1)
        x = self.SDU5(x)
        x = F.max_pool2d(x,3,2,1)
        # HAP (n , c ,h , w) = (batch_size, 320, 4,  1)
        x = F.avg_pool2d(x,(1,4),1)
        x = self.SN(x) 
        feat = x.view(x.size(0), -1) 
        x = self.classifier(feat)
        return x,feat

from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.nn import init
from dataset import NBR_ID,NBR_MODELS,NBR_COLORS

#NBR_ID = 85288 
#NBR_MODELS = 250
#NBR_COLORS = 7


__all__ = ['ResNet' , 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101:torchvision.models.resnet101,
        152:torchvision.models.resnet152,
    }
    
    def __init__(self,
                 depth, 
                 pretrained=True, 
                 num_features=1024,  
                 num_id=NBR_ID, 
                 num_model=NBR_MODELS, 
                 num_color=NBR_COLORS):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.num_features = num_features
        self.num_id = num_id
        self.num_model = num_model
        self.num_color = num_color 

        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=self.pretrained)

        out_planes = self.base.fc.in_features

        self.feat = nn.Linear(out_planes, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias,0)
        init.constant(self.feat_bn.weight,1)
        init.constant(self.feat_bn.bias,0)
        
        if self.num_id > 0:
            self.id_classifier = nn.Linear(self.num_features, self.num_id)
            init.normal(self.id_classifier.weight, std=0.001)
            init.constant(self.id_classifier.bias, 0)
        if self.num_model > 0:
            self.model_classifier = nn.Linear(self.num_features, self.num_model)
            init.normal(self.model_classifier.weight, std=0.001)
            init.constant(self.model_classifier.bias, 0)
        if self.num_id > 0:
            self.color_classifier = nn.Linear(self.num_features, self.num_color)
            init.normal(self.color_classifier.weight, std=0.001)
            init.constant(self.color_classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self,x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = F.avg_pool2d(x,x.size()[2:])
        x = x.view(x.size(0), -1)
        
        x = self.feat(x)
        x = self.feat_bn(x)
        x = F.relu(x)

        if self.num_id > 0:
            f_id = self.id_classifier(x)
        if self.num_color > 0:
            f_color = self.color_classifier(x)
        if self.num_model > 0:
            f_model = self.model_classifier(x)

        return f_model,f_color,f_id,x


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight,1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

def resnet18(**kwargs):
    return ResNet(18,**kwargs)
    
def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet152(**kwargs):
    return ResNet(152, **kwargs)





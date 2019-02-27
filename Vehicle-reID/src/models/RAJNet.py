#ResNet-Attributes-J Net
from __future__ import absolute_import
from torch import nn
import torch
from torch.nn import functional as F
import torchvision
from torch.nn import init
from dataset import NBR_ID, NBR_MODELS, NBR_COLORS

class RAJNet(nn.Module):
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
                 num_features = 1024,
                 dropout=0,
                 num_id=NBR_ID, 
                 num_model=NBR_MODELS,
                 num_color=NBR_COLORS):
        super(RAJNet, self).__init__()
        #print("RAJNet")
        self.depth = depth
        self.pretrained = pretrained
        
    
        if depth not in RAJNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = RAJNet.__factory[depth](pretrained=self.pretrained)

        
        self.num_features = num_features
        self.dropout = dropout
        #self.has_embedding = base_dim > 0
        self.num_id = num_id
        self.num_model = num_model
        self.num_color = num_color

        base_dim  = self.base.fc.in_features

        #if self.has_embedding:
       
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.num_id > 0:    
            self.id_fc_1 = nn.Linear(base_dim, 2048)
            self.id_fc_2 = nn.ReLU(inplace=True)
            self.id_fc_3 = nn.Linear(2048, 1024)
            self.id_fc_4 = nn.ReLU(inplace=True)
            self.id_branch = nn.Sequential(
                self.id_fc_1,
                self.id_fc_2,
                self.id_fc_3,
                self.id_fc_4,
            )
            self.id_classifier = nn.Linear(1024, self.num_id)
            init.normal(self.id_classifier.weight, std=0.001)
            init.constant(self.id_classifier.bias, 0)
        if self.num_model > 0:
            self.model_fc_1 = nn.Linear(base_dim, 1024)
            self.model_fc_2 = nn.ReLU(inplace=True)
            self.model_fc_3 = nn.Linear(1024, 512)
            self.model_fc_4 = nn.ReLU(inplace=True)
            self.model_branch = nn.Sequential(
                self.model_fc_1,
                self.model_fc_2,
                self.model_fc_3,
                self.model_fc_4,
            )            
            self.model_classifier = nn.Linear(512, self.num_model)
            init.normal(self.model_classifier.weight, std=0.001)
            init.constant(self.model_classifier.bias, 0)
        if self.num_color > 0:
            self.color_fc_1 = nn.Linear(base_dim, 1024)
            self.color_fc_2 = nn.ReLU(inplace=True)
            self.color_fc_3 = nn.Linear(1024,512)
            self.color_fc_4 = nn.ReLU(inplace=True)
            self.color_branch = nn.Sequential(
                self.color_fc_1,
                self.color_fc_2,
                self.color_fc_3,
                self.color_fc_4,
            )
            self.color_classifier = nn.Linear(512, self.num_color)
            init.normal(self.color_classifier.weight, std=0.001)
            init.constant(self.color_classifier.bias, 0)
        
        self.feat = nn.Linear(2048,1024)
    
        if not self.pretrained:
            self.reset_params()

    def forward(self,x):
        
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        x = F.avg_pool2d(x,x.size()[2:])
        x = x.view(x.size(0), -1)
        
        if self.num_id > 0:
            id_branch = self.id_branch(x)
            id_output = self.id_classifier(id_branch)
        if self.num_color > 0:
            color_branch = self.color_branch(x)
            color_output = self.color_classifier(color_branch)
        if self.num_model > 0:
            model_branch = self.model_branch(x)
            model_output = self.model_classifier(model_branch)
        combined_feat = torch.cat((id_branch,color_branch,model_branch), dim=1)
        #print(combined_feat.size(0))
        total_output = self.feat(combined_feat)
        #print(id_output.size)
        #return id_output,color_output,model_output,total_output
        return model_output, color_output, id_output, total_output

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

def RAJNet18(**kwargs):
    return RAJNet(18, **kwargs)
    
def RAJNet34(**kwargs):
    return RAJNet(34, **kwargs)

def RAJNet50(**kwargs):
    return RAJNet(50, **kwargs)

def RAJNet101(**kwargs):
    return RAJNet(101, **kwargs)

def RAJNet152(**kwargs):
    return RAJNet(152, **kwargs)

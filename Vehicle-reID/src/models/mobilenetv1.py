import torch
import torch.nn as nn
import torch.nn.functional as F

NBR_ID = 85288
class mobilenetv1(nn.Module):
    def __init__(self):
        super(mobilenetv1,self).__init__()
        
        def conv_1x1_bn(inp,oup):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
    
        def conv_dw(inp,oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
        
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.conv1_1x1 = conv_1x1_bn(128,512)
        self.conv_1x1_2 = conv_1x1_bn(512,512) 
        self.conv2_1x1 = conv_1x1_bn(256,512)
        self.conv3_1x1 = conv_1x1_bn(512,512)
        
        self.conv1 = conv_bn( 3, 32, 2)
        self.conv_pw_1 = conv_dw(32,64,1)
        self.conv_pw_2 = conv_dw(64,128,2)
        self.conv_pw_3 = conv_dw(128,128,1)
        self.conv_pw_4 = conv_dw(128,256,2)
        self.conv_pw_5 = conv_dw(256,256,1)
        self.conv_pw_6 = conv_dw(256,512,2)
        self.conv_pw_7 = conv_dw(512,512,1)
        self.conv_pw_8 = conv_dw(512,512,1)
        self.conv_pw_9 = conv_dw(512,512,1)
        self.conv_pw_10= conv_dw(512,512,1)
        self.conv_pw_11= conv_dw(512,512,1)
        self.conv_pw_12= conv_dw(512,1024,2)
        self.conv_pw_13= conv_dw(1024,1024,1)
        
        self.f_acs_new = nn.Linear(2560,1024)
        self.predictions_id_new = nn.Linear(1024,NBR_ID)

         
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_pw_1(x)
        x = self.conv_pw_2(x)
        local_56 = self.conv_pw_3(x)
        x = self.conv_pw_4(local_56)
        local_28 = self.conv_pw_5(x)
        x = self.conv_pw_6(local_28)
        x = self.conv_pw_7(x)
        x = self.conv_pw_8(x)
        x = self.conv_pw_9(x)
        x = self.conv_pw_10(x)
        local_14 = self.conv_pw_11(x)
        x = self.conv_pw_12(local_14)
        f_base1 = self.conv_pw_13(x)
        f_base = F.avg_pool2d(f_base,1024)
    
        local_56_1 = self.conv1_1x1(local_56)
        local_56_2 = self.conv_1x1_2(local_56_1)
        local_28_1 = self.conv2_1x1(local_28)
        local_28_2 = self.conv_1x1_2(local_28_1)
        local_14_1 = self.conv3_1x1(local_14)
        local_14_2 = self.conv_1x1_2(local_14_1)

        local_56_f = F.max_pool2d(local_56_2,512)
        local_28_f = F.max_pool2d(local_28_2,512)
        local_14_f = F.max_pool2d(local_14_2,512)
        #attention!!!
        f_base = torch.cat((f_base,local_56_f, local_28_f, local_14_f),2)
        
        f_acs = self.f_acs_new(f_base)
        f_id =  self.predictions_id_new(f_acs)
        
        
        return f_id











         

        


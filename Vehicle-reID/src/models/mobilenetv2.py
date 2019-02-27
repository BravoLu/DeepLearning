import torch
import torch.nn as nn
import torch.nn.functional as F

NBR_MODELS = 250
NBR_COLORS = 7
NBR_ID = 85288

class mobilenetv2(nn.Module):
    def __init__(self):
        super(mobilenetv2,self).__init__()
        
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

        #self.conv1_1x1 = conv_1x1_bn(128,512)
        #self.conv_1x1_2 = conv_1x1_bn(512,512) 
        #self.conv2_1x1 = conv_1x1_bn(256,512)
        #self.conv3_1x1 = conv_1x1_bn(512,512)
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

        self.base = nn.Sequential(
              self.conv1,
              self.conv_pw_1,
              self.conv_pw_2,
              self.conv_pw_3,
              self.conv_pw_4,
              self.conv_pw_5,
              self.conv_pw_6,
              self.conv_pw_7,
              self.conv_pw_8,
              self.conv_pw_9,
              self.conv_pw_10,
              self.conv_pw_11,
              self.conv_pw_12,
              self.conv_pw_13,
        )

        #self.f_acs_new = nn.Linear(2560,1024)
        #self.predictions_id_new = nn.Linear(1024,NBR_ID)
        self.f_acs = nn.Linear(1024,1024)
        self.f_model = nn.Linear(1024, NBR_MODELS)
        self.f_color = nn.Linear(1024, NBR_COLORS)
        self.f_id    = nn.Linear(1024, NBR_ID)        
 
    def forward(self, x):
        #x = self.conv1(x)
        #x = self.conv_pw_1(x)
        #x = self.conv_pw_2(x)
        #x = self.conv_pw_3(x)
        #x = self.conv_pw_4(x)
        #x = self.conv_pw_5(x)
        #x = self.conv_pw_6(x)
        #x = self.conv_pw_7(x)
        #x = self.conv_pw_8(x)
        #x = self.conv_pw_9(x)
        #x = self.conv_pw_10(x)
        #x = self.conv_pw_11(x)
        #x = self.conv_pw_12(x)
        #x = self.conv_pw_13(x)
        x = self.base(x)
        x = F.avg_pool2d(x, (7,7))
        x = x.view(x.size(0), -1)
        x = self.f_acs(x)
        model = self.f_model(x)
        #model = F.softmax(model)
        color = self.f_color(x)
        #color = F.softmax(color)
        ID    = self.f_id(x)
        #ID    = F.softmax(ID)

        return model,color,ID













         

        


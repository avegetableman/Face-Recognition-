import torch.nn as nn
def bn(in_c,out_C,stride):
    return nn.Sequential(
        nn.Conv2d(in_c,out_C,3,stride,1,bias=False),
        nn.BatchNorm2d(out_C),
        nn.ReLU6()
    )
def dw(in_c,out_C,stride):
    model = nn.Sequential(
        nn.Conv2d(in_c,in_c,3,stride,1,groups=in_c),
        nn.BatchNorm2d(in_c),
        nn.ReLU6(),

        nn.Conv2d(in_c,out_C,1,1,0,bias=False),
        nn.BatchNorm2d(out_C),
        nn.ReLU6()
    )
    return model
class MobilenetV1(nn.Module):
    def __init__(self):
        super(MobilenetV1,self).__init__()
        self.target1 = nn.Sequential(
            bn(3,32,2),
            dw(32,64,1),
            dw(64,128,2),
            dw(128,128,1),
            dw(128,256,2),
            dw(256,256,1)
        )
        self.target2 = nn.Sequential(
            dw(256,512,2),
            dw(512,512,1),
            dw(512,512,1),
            dw(512,512,1),
            dw(512,512,1),
            dw(512,512,1),
        )
        self.target3 = nn.Sequential(
            dw(512,1024,2),
            dw(1024,1024,1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(1024,10575)
        self.log=nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.target1(x)
        x = self.target2(x)
        x = self.target3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x =self.log(x)
        return x
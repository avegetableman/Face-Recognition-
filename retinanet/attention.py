import torch
import torch.nn as nn
import math
class Channel_attention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(Channel_attention, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, False),
        )
        self.sig =nn.Sigmoid()
    def forward(self, x):
        b,c,h,w = x.size()
        avg = self.avg(x).view([b,c])
        max = self.max(x).view([b,c])
        avg = self.fc(avg)
        max = self.fc(max)
        out = self.sig(avg+max).view([b,c,1,1])
        return out

class Spatial_attention(nn.Module):
    def __init__(self,k=3):
        super(Spatial_attention, self).__init__()
        self.conv = nn.Conv2d(2,1,k,1,k//2)
        self.sig=nn.Sigmoid()
    def forward(self, x):
        max_x,_ = torch.max(x,dim=1,keepdim=True)
        avg_x = torch.mean(x,dim=1,keepdim=True)
        ans = torch.cat([max_x,avg_x],dim=1)
        target = self.conv(ans)
        return self.sig(target)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction,False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels,False),
            nn.Sigmoid()
        )
    def float(self,x):
        b,c,w,h = x.size()
        avg = self.avg(x).view([b,c])
        an = self.fc(avg).view([b,c,w,h])
        return an*x

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel = Channel_attention(channels,reduction)
        self.spatial = Spatial_attention()
    def forward(self, x):
        x = x*self.channel(x)
        x = x*self.spatial(x)
        return x

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.channel = eca_block(channels)
        self.spatial = Spatial_attention()
    def forward(self, x):
        x = self.channel(x)
        x = x*self.spatial(x)
        return x

if __name__ == "__main__":
    test = Spatial_attention()
    seed = torch.rand(1,3,224,224)
    print(test(seed))
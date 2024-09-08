import torch.nn as nn
import numpy as np
import math
class Anchors(nn.Module):
    def __init__(self):
        super(Anchors, self).__init__()
        self.layer=[3,4,5,6,7]
        self.scale=[0.5,1,2]
        self.ratial = [2.0**0,2.0**(1.0/3.0),2.0**(2.0/3.0)]
        self.stride = [2**x for x in self.layer]
        self.size = [2**(x+2) for x in self.layer]
    def forward(self,features,layer):
        batch = features.shape[0]
        feature_size=features.shape[2]
        anchors=generate_anchor(batch,self.size[layer],self.stride[layer],feature_size,self.ratial,self.scale)
        return anchors
def generate_anchor(batch,size,stride,feature_size,ratial,scale):
    anchor=[]
    for i in range(batch):
        temp=[]
        row=0
        area=size**2
        while row<feature_size:
            col=0
            while col<feature_size:
                for j in ratial:
                    area=area*j
                    for s in scale:
                        h=math.sqrt(area)/s
                        w=s*h
                        x1=col
                        y1=row
                        x=x1-s*w
                        y=y1-s*h
                        temp.append([x,y,w,h])
                col=col+stride
            row=row+stride
        anchor.append(temp)
    return anchor

if __name__ == "__main__":
    import torch
    seed=torch.rand(1,256,224,224)
    a=Anchors()
    #print(torch.tensor([a(seed,x) for x in range(4)]).shape)
    print(torch.cat([torch.tensor(a(seed,x)) for x in range(5)],dim=1).shape) 
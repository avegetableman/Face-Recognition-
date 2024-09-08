from mobilenetv1 import MobilenetV1
import torch.nn as nn
import torch
class Facenet(nn.Module):
    def __init__(self,mode='train',embedding_size=128,num_class=None,drop=0.5,pretrained=False):
        super(Facenet, self).__init__()
        if pretrained:
            self.mobilenet=torch.load('mobilenet.bin')
        else:
            self.mobilenet = MobilenetV1()
        del self.mobilenet.avg_pool
        del self.mobilenet.fc
        del self.mobilenet.log
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(1024,embedding_size)
        self.drop=nn.Dropout(drop)
        self.norm = nn.BatchNorm1d(embedding_size)
        if mode == 'train':
            self.classfier = nn.Linear(embedding_size,num_class)
            self.log=nn.LogSoftmax(dim=1)
    def forward(self, x,mode='predict'):
        x = self.mobilenet.target1(x)
        x = self.mobilenet.target2(x)
        x = self.mobilenet.target3(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        v_x = nn.functional.normalize(x,p=2,dim=1)
        x = self.drop(x)
        x = self.norm(x)
        cls = self.classfier(x)
        cls  = self.log(cls)
        if mode == 'train':
            return v_x,cls
        else:
            return v_x
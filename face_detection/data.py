from torch.utils.data import dataset
from PIL import Image
import os
import torch.nn as nn
import random
import math
class Data_tacker(dataset.Dataset):
    def __init__(self,path,transform):
        super(Data_tacker,self).__init__()
        self.path = path
        self.length = len(os.listdir(self.path))
        self.dir = os.listdir(self.path)
        self.transform=transform
    def __getitem__(self, index):
        temp = random.randint(0, self.length - 1)  # 确保temp不会等于self.length
        if temp == index:
            # 如果index接近开始或结束，选择另一个方向的偏移
            if index < self.length - 2:
                temp = index + 2
            elif index > 1:
                temp = index - 2
            else:
                # 如果index是0或1，选择1作为temp
                temp = 1
        picture = os.listdir(os.path.join(self.path,self.dir[index]))
        picture_index = random.sample(range(len(picture)),k=2)
        anchor = self.transform(Image.open(os.path.join(self.path,self.dir[index],picture[picture_index[0]])))
        positive = self.transform(Image.open(os.path.join(self.path,self.dir[index],picture[picture_index[1]])))
        negative_picture = os.listdir(os.path.join(self.path,self.dir[temp]))
        negative_picture_index = random.randint(0,len(negative_picture)-1)
        negative = self.transform(Image.open(os.path.join(self.path,self.dir[temp],negative_picture[negative_picture_index])))
        return (anchor,positive,negative,index,temp)
    def __len__(self):
        return self.length
        #count=0
        #for i in self.dir:
         #   count += len(os.listdir(os.path.join(self.path,i)))
        #return count
if __name__ == '__main__':
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import transforms
    trans=transforms.Compose([transforms.Resize([130,130]),transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)])
    a=Data_tacker(r"E:\6\AI\dataset\datasets",transform=trans)
    from torch.utils.data import DataLoader
    q=DataLoader(a,batch_size=12)
    print(next(iter(q)))
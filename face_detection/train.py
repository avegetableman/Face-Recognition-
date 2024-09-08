import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.rich import tqdm
from facenet import Facenet
from data import Data_tacker
import matplotlib.pyplot as plt
from tensorboardX  import SummaryWriter
write = SummaryWriter('./face')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path =r"E:/6/AI/dataset/datasets"
from torchvision.transforms import functional as F

def to_three_channel(image):
    if image.mode != 'RGB':
        # 将灰度图像转换为三通道
        image = F.to_tensor(image)  # 将图像转换为一个单通道的张量
        image = image.expand(3, -1, -1)  # 扩展到三个通道
        image = F.to_pil_image(image)  # 将张量转换回PIL图像
    return image

# 应用这个函数到你的转换流程中
trans = transforms.Compose([
    transforms.Resize([130, 130]),
    to_three_channel,  # 使用自定义的转换函数
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
#trans=transforms.Compose([transforms.Resize([130,130]),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)])
data = Data_tacker(path,transform=trans)
train_loader = DataLoader(data,batch_size=64,shuffle=True,drop_last=True)

model = Facenet(num_class=10575).to(device)
cer = torch.nn.TripletMarginLoss(margin=1.0)
nll_cer = torch.nn.NLLLoss()
optim =  torch.optim.Adam(model.parameters(),lr=0.0002)

epoch = 800
loss_sum=[]
for i in tqdm(range(epoch),unit='epochs',desc='training'):
    temp0=0
    for j,(anchor,positive,negative,index,temp) in enumerate(train_loader):
        anchor_x,anchor_cls = model(anchor.to(device),mode='train')
        positive_x,positive_cls = model(positive.to(device),mode='train')
        negative_x,negative_cls = model(negative.to(device),mode='train')
        cls_loss=(nll_cer(anchor_cls,index.to(device))+nll_cer(positive_cls,index.to(device))+nll_cer(negative_cls,temp.to(device)))/3
        tri_loss = cer(anchor_x,positive_x,negative_x).to(device)
        loss = cls_loss+tri_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        temp0+=loss.item()
    loss_sum.append(temp0/(j+1))
    write.add_scalar('loss',(temp0/(j+1)),i)
plt.plot(range(epoch),loss_sum)
plt.savefig('loss.jpg')
torch.save(model,'face.bin')
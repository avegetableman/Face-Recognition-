import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from tqdm.rich import tqdm
from mobilenetv1 import MobilenetV1
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
writer = SummaryWriter('./t')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path =r"E:/6/AI/dataset/datasets"
trans=transforms.Compose([transforms.Resize([160,160]),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)])
data = ImageFolder(root=path,transform=trans)
train_loader = DataLoader(data,batch_size=64,shuffle=True,drop_last=True)

model = MobilenetV1().to(device)
nll_cer = torch.nn.NLLLoss()
optim =  torch.optim.Adam(model.parameters(),lr=0.0002)

epoch = 30
sloss=[]
for i in tqdm(range(epoch),unit='epochs',desc='training'):
    tem=0
    for j,(img,lable) in enumerate(train_loader):
        img,lable = img.to(device),lable.to(device)
        optim.zero_grad()
        cls = model(img)
        loss = nll_cer(cls,lable)
        loss.backward()
        optim.step()
        tem+=loss.item()
    sloss.append(tem/(j-1))
    writer.add_scalar('loss',tem,i)
plt.plot(range(epoch),sloss)
plt.savefig("mobloss.jpg")
torch.save(model,'mobilenet.bin')
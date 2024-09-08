import torch
import torchvision.transforms as transform
from data import CelebA
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from model import Retinanet_with_attention
from loss import call_iou,Foacal_loss,MakeLabel
import torch.nn as nn
import tensorboardX
from anchors import Anchors
import numpy as np
from utils import transform_box
from PIL import Image,ImageDraw

unt = transform.ToPILImage()
writer = tensorboardX.SummaryWriter(log_dir='./runs')

epochs = 30
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trans = transform.Compose([transform.Resize((224,224)),transform.ToTensor()])
data = CelebA(r'D:\data\CelebA\Img\img_celeba\img_celeba','list_bbox_celeba.txt',trans=trans,resize=(224,224))
train_data = DataLoader(dataset=data,batch_size=128,pin_memory=True)

model = Retinanet_with_attention().to(device)
optim = torch.optim.Adam(model.parameters(),lr=0.0002)
focal_loss = Foacal_loss()
regress = nn.SmoothL1Loss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',patience=5)
LabelMake = MakeLabel()
anchor = Anchors()

def make_regress_label(anchornum, label):
    target = []
    for anchor in label:
        tmp = np.tile(anchor.detach().cpu().numpy(), (anchornum, 1))
        target.append(tmp)
    
    # 将整个 target 列表转换为单个 numpy.ndarray，然后再转换为 torch.Tensor
    target = np.array(target)
    return torch.tensor(target)

for epoch in tqdm(range(epochs),desc='training:',unit='epoch'):
    for i,dataX in enumerate(train_data):
        img,label = dataX #[batch,4]
        img, label = img.to(device), label.to(device)
        pred_regress,pred_cls = model(img)
        pred_cls = pred_cls.to(device)
        pred_regress = pred_regress.to(device)
        generated = anchor(img).repeat(img.shape[0],1,1).to(device)
        new_label = LabelMake(generated,label).to(device) #[batch,anchor,1]
        class_loss = focal_loss(pred_cls,new_label).to(device)
        regress_label = make_regress_label(generated.shape[1],label).to(device)
        regress_loss = regress(pred_regress*new_label,regress_label*new_label-generated*new_label).to(device)
        total_loss = regress_loss+class_loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()
    answer = (pred_regress[0].unsqueeze(0),pred_cls[0].unsqueeze(0))
    t = transform_box(Anchors)
    a = t(answer)
    image = unt(img[0])
    draw = ImageDraw.Draw(image)
    for ii in a:
        x,y,w,h = ii['zb']
        box = [x,y,x+w,y+h]
        draw.rectangle(box, outline="red", width=2)
    image.save('./picture/detect{epoch}.png'.format(epoch=epoch))
    scheduler.step(total_loss)
    writer.add_scalar('Loss/Train', total_loss.item()/(i+1), epoch)
    writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], epoch)

writer.close()
torch.save(model,'detect_with_attention.bin')
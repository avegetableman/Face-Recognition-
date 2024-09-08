from torch.utils.data import Dataset
import re
from PIL import Image
import os
import torch
class CelebA(Dataset):
    def __init__(self,path,txt,trans,resize):
        super().__init__()
        self.path=path
        self.txt=txt
        self.n = re.compile(r'(.*?)\n')
        with open(self.txt,'r') as f:
            self.data=f.readlines()
        self.trans=trans
        self.wid,self.hig=resize
    def __getitem__(self, index):
        new_index=index+2
        img=[x for x in self.n.findall(self.data[new_index])[0].split(' ') if x != '']
        path = os.path.join(self.path,img[0])
        
        #new image
        image = Image.open(path)
        width,height=image.size
        old=max(width,height)
        neww=max(self.wid,self.hig)
        sqr=max(width,height)
        background=Image.new(mode='RGB',size=(sqr,sqr),color=(0,0,0))
        background.paste(image,(0,0))
        new_image=self.trans(background)
        sY=neww/old
        x=int(img[1])*sY
        y=int(img[2])*sY
        w=int(img[3])*sY
        h=int(img[4])*sY
        label=torch.tensor([x,y,w,h])
        return (new_image,label)
    def __len__(self):
        return int(self.n.findall(self.data[0])[0])

if __name__ =="__main__":
    from torchvision.transforms import transforms
    from PIL import Image, ImageDraw
    trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    unload=transforms.ToPILImage()
    a=CelebA(r'G:\6\AI\dataset\CelebA\Img\img_celeba\img_celeba','list_bbox_celeba.txt',trans=trans,resize=(224,224))
    image,label=a[5]
    from loss import MakeLabel
    from anchors import Anchors
    a = Anchors().cuda()
    seed = torch.rand(1,256,224,224).cuda()
    anchor=a(seed).repeat(2,1,1)
    label = label.unsqueeze(0).cuda()
    image = image.unsqueeze(0).cuda()
    ml=MakeLabel().cuda()
    label_=ml(anchor,label,cls_label=0,labelnum=1)
    print(label_.shape)
    for i in label_[0]:
        if i[0] == 1:
            print(6)
    '''print(image.shape)
    x,y,w,h=label
    strat=(int(x),int(y))
    end=(int(w)+int(x),int(h)+int(y))
    nnimg=unload(image)
    draw = ImageDraw.Draw(nnimg)
    xy=[strat,end]
    draw.rectangle(xy,outline='green')
    nnimg.show()'''
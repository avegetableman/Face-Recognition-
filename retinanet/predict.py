import torch
from model import Retinanet
import torchvision.transforms as transform
from utils import transform_box
from PIL import Image,ImageDraw
from anchors import Anchors
trans = transform.Compose([transform.ToTensor(),transform.Resize((224,224))])
unt = transform.ToPILImage()
model = torch.load('detect.bin').cuda()
picture = Image.open('hzy.jpg')
image = trans(picture).unsqueeze(0).cuda()
answer = model(image)
t = transform_box(Anchors)
a = t(answer,th=0.5)
b,c=answer
#print(torch.max(c,dim=1))
image = unt(image[0])
draw = ImageDraw.Draw(image)
for i in a:
    x,y,w,h = i['zb']
    box = [x,y,x+w,y+h]
    draw.rectangle(box, outline="red", width=3)
image.save('detect.png')

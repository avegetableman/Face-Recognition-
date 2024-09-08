import torch
from onx import QuantizableModel
net=torch.load('face.bin',map_location='cpu').eval()
f1=r"E:/6/AI/dataset/datasets/0000045/001.jpg"
f2=r"E:/6/AI/dataset/datasets/0000045/008.jpg"
f3=r"E:/6/AI/dataset/datasets/0000045/009.jpg"
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from PIL import Image
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
a1=net(trans(Image.open(f1)).unsqueeze(0),mode='predict')
a2=net(trans(Image.open(f2)).unsqueeze(0),mode='predict')
a3=net(trans(Image.open(f3)).unsqueeze(0),mode='predict')
print(torch.nn.functional.pairwise_distance(a1, a2, p=2.0, eps=1e-6, keepdim=False))
print(torch.nn.functional.pairwise_distance(a3, a2, p=2.0, eps=1e-6, keepdim=False))

import onnxruntime  as ort
f1=r"E:/6/AI/dataset/datasets/0000045/001.jpg"
f2=r"E:/6/AI/dataset/datasets/0000045/008.jpg"
f3=r"E:/6/AI/dataset/datasets/0000045/009.jpg"
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
def to_three_channel(image):
    if image.mode != 'RGB':
        # 将灰度图像转换为三通道
        image = F.to_tensor(image)  # 将图像转换为一个单通道的张量
        image = image.expand(3, -1, -1)  # 扩展到三个通道
        image = F.to_pil_image(image)  # 将张量转换回PIL图像
    return image
trans = transforms.Compose([
    transforms.Resize([130, 130]),
    to_three_channel, 
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
a1=trans(Image.open(f1)).unsqueeze(0).detach().numpy()
a2=trans(Image.open(f2)).unsqueeze(0).detach().numpy()
a3=trans(Image.open(f3)).unsqueeze(0).detach().numpy()
pro = ["CPUExecutionProvider"]
a= ort.InferenceSession('face.onnx',providers=pro)
ort_inputs = {a.get_inputs()[0].name: a1}
ort_inputs1 = {a.get_inputs()[0].name: a2}
ort_inputs2 = {a.get_inputs()[0].name: a3}
o=a.run(output_names=['output'],input_feed=ort_inputs)
o1=a.run(output_names=['output'],input_feed=ort_inputs1)
o2=a.run(output_names=['output'],input_feed=ort_inputs2)
print(np.linalg.norm(o2[0] - o1[0]))
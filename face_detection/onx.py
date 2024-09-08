import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# 定义一个带量化stub的模型
class QuantizableModel(nn.Module):
    def __init__(self, model):
        super(QuantizableModel, self).__init__()
        self.model = model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model.mobilenet.target1(x)
        x = self.model.mobilenet.target2(x)
        x = self.model.mobilenet.target3(x)
        x = self.dequant(x)
        return x
net = torch.load('face.bin', map_location='cpu')
net.eval()
quantizable_net = QuantizableModel(net)
quantizable_net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(quantizable_net, inplace=True)
path =r"E:/6/AI/dataset/datasets"
trans=transforms.Compose([transforms.Resize([130,130]),transforms.ToTensor(),transforms.Normalize(mean=0.5,std=0.5)])
data = ImageFolder(root=path,transform=trans)
train_loader = DataLoader(data,batch_size=64,shuffle=True,drop_last=True)
print(next(iter(train_loader))[0].shape)
quantizable_net(next(iter(train_loader))[0])
quantized_model = torch.quantization.convert(quantizable_net, inplace=True)
torch.save(quantized_model,'int8.bin')
dummy_input = torch.randn(1, 3, 130, 130, requires_grad=True)
'''torch.onnx.export(quantized_model, dummy_input, 'face.onnx', 
                  input_names=['input'], output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})'''

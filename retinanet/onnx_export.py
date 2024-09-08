from model import Retinanet
import torch
model = torch.load('detect.bin',map_location='cpu')
model.eval()
inputs = torch.rand(1,3,224,224)
torch.onnx.export(
    model,
    inputs,
    'detect.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output','output2'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'},'output2': {0: 'batch_size'}}
)
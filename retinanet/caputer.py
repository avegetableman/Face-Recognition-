import cv2
from torchvision import transforms
from model import Retinanet
from utils import transform_box
import torch
from anchors import Anchors
from PIL import Image
from utils import NMS
filter = NMS()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('detect.bin').to(device)
trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
unload = transforms.ToPILImage()
#font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (255, 255, 255)  # 白色
font_thickness = 2


def crop(image):
    height, width, _ = image.shape
    square_size = min(width, height)
    x_center = width // 2
    y_center = height // 2
    x_start = x_center - square_size // 2
    y_start = y_center - square_size // 2
    cropped_image = image[y_start:y_start + square_size, x_start:x_start + square_size]
    return cropped_image
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
while True:
    flag, image = cap.read()
    if not flag:
        break
    data0 = crop(image)
    convert = cv2.cvtColor(data0, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(convert)
    data = trans(image_pil)
    data = data.unsqueeze(0).to(device)
    answer = model(data)
    t = transform_box(Anchors)
    a = t(answer,th=0.5)
    annn = filter(a)
    for i in annn:
        x,y,w,h = i['box']
        cv2.putText(data0, str(round(float(i['confidence']),3)), (int(x),int(y)), font, font_scale, font_color, font_thickness)
        cv2.rectangle(data0, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    cv2.imshow('face', data0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

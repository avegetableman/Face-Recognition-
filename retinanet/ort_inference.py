import onnxruntime as ort
import numpy as np
from utils_for_onnx import NMS,Anchors,transform_box
import cv2
filter = NMS()

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

def resize(image,size=(224,224)):
    if image.shape[1] == 224 & image.shape[0] == 224:
        return image
    else:
        return cv2.resize(image,size)

path = 'detect.onnx'
session = ort.InferenceSession(path)
input_name = session.get_inputs()[0].name
output_names = [session.get_outputs()[0].name, session.get_outputs()[1].name]
image = cv2.imread('hzy.jpg').astype(np.float32) / 255.0
data0 = crop(image)
data0 = resize(data0)
convert = cv2.cvtColor(data0, cv2.COLOR_BGR2RGB)
e = np.expand_dims(convert, axis=0)
image_bchw = np.transpose(e, (0, 3, 1, 2))
answer = session.run(output_names, {input_name: image_bchw})
t = transform_box(Anchors)
a = t.forward(answer,th=0.5)
annn = filter.forward(a)
for i in annn:
    x,y,w,h = i['box']
    cv2.putText(data0, str(round(float(i['confidence']),3)), (int(x),int(y)), font, font_scale, font_color, font_thickness)
    cv2.rectangle(data0, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
cv2.imwrite('face_onnx.jpg', data0*255)


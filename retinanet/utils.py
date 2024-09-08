import torch.nn as nn
import torch
def iou(box0,box1):#loss中的计算iou方法是为了对应celeba数据集的，这里进行重写用于计算俩坐标   慢的话尝试用tensor操作
    x,y,w,h =box0
    x1,y1,w1,h1 = box1
    x0 = x+w
    y0=y+h
    x11 = x1+w1
    y11 = y1+h1
    ax = max(x,x1)
    ay = max(y,y1)
    a1x = min(x0,x11)
    a1y = min(y0,y11)
    width = max(a1x - ax,0)
    height = max(a1y - ay,0)
    inner_area = width*height
    union = max(w*h + w1*h1 - inner_area,1e-6)
    iou = inner_area / (union)
    return iou
def is_no_zero(x):
    target=[]
    for i in x:
        if i.sum() > 0:
            target.append(torch.ones(i.shape).detach().cpu().numpy())
        else:
            target.append(torch.zeros(i.shape).detach().cpu().numpy())
    return torch.tensor(target)
def get_indices(x):
    indices = [] 
    for index,i in enumerate(x):
        if i.sum() > 0:
            indices.append(index)
    return indices

class transform_box(nn.Module):
    def __init__(self,anchor,size=(224,224)):
        super(transform_box, self).__init__()
        self.anchor = anchor()
        self.seed = self.anchor(torch.rand(1,3,size[0],size[1]))
    def forward(self,answer,th=0.4): #only one batch and cpu
        zb_answer,cls_answer = answer
        cls_answer = cls_answer.detach().cpu()[0]
        zb_answer = zb_answer.detach().cpu()[0]
        index = cls_answer > th
        index2=index.repeat(1,4)
        zb_answer = zb_answer[index2]
        an = self.seed[0][index2]
        cls_answer = cls_answer[index]
        target = []
        for i in range(cls_answer.shape[0]):
            temp={}
            temp['cls'] = cls_answer[i]
            temp['zb'] = an[i*4]+zb_answer[i*4],an[i*4+1]+zb_answer[i*4+1],an[i*4+2]+zb_answer[i*4+2],an[i*4+3]+zb_answer[i*4+3]
            target.append(temp)
        return target

class NMS(nn.Module):
    def __init__(self, score_threshold=0.3, nms_threshold=0.5):
        super(NMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def forward(self, answer): # only one batch and cpu
        target = []
        index = {}
        a = 0
        for i in answer:
            index[i['cls']] = a
            a += 1
        new_index = sorted(index.items(), reverse=True)
        new_index = [i for i in new_index if i[0] > self.score_threshold]
        for n in range(len(new_index)):
            if new_index[n] is None:
                continue

            temp = n + 1
            for n_in in range(temp, len(new_index)):
                if new_index[n_in] is not None and iou(answer[new_index[n][1]]['zb'], answer[new_index[n_in][1]]['zb']) > self.nms_threshold:
                    new_index[n_in] = None
        target_index = [x for x in new_index if x is not None]
        for q in range(len(target_index)):
            target_dict = {}
            target_dict['confidence'] = target_index[q][0]
            target_dict['box'] = answer[target_index[q][1]]['zb']
            target.append(target_dict)

        return target

if __name__ == '__main__':
    from anchors import Anchors
    a = transform_box(Anchors)
    x = (torch.rand(1,9441,4),torch.rand(1,9441,1))
    y = a(x)
    print(y)
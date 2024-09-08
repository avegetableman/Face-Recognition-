import torch.nn as nn
import torch.nn.functional as F
import torch
def call_iou(anchor, label):
    # 确保 label 的形状与 anchor 一致
    label = label.unsqueeze(1).repeat(1, anchor.shape[1], 1)
    anchor_widths = anchor[:, :, 2]
    anchor_heights = anchor[:, :, 3]
    anchor_widths = torch.clamp(anchor_widths, max=1000)
    anchor_heights = torch.clamp(anchor_heights, max=1000)
    x1 = torch.max(anchor[:, :, 0], label[:, :, 0])
    y1 = torch.max(anchor[:, :, 1], label[:, :, 1])
    x2 = torch.min(anchor[:, :, 0] + anchor_widths, label[:, :, 0] + label[:, :, 2])
    y2 = torch.min(anchor[:, :, 1] + anchor_heights, label[:, :, 1] + label[:, :, 3])
    inter_width = torch.clamp(x2 - x1, min=0)
    inter_height = torch.clamp(y2 - y1, min=0)
    inter_area = inter_width * inter_height
    anchor_area = anchor_widths * anchor_heights
    label_area = label[:, :, 2] * label[:, :, 3]
    union_area = torch.clamp(anchor_area + label_area - inter_area, min=1e-6)
    iou = inter_area / union_area

    return iou





class Foacal_loss(nn.Module):
    def __init__(self,gamma=2,ap=0):
        super(Foacal_loss, self).__init__()
        self.gamma = gamma
        self.ap=ap
        
    def forward(self,pred,label):
        loss = F.binary_cross_entropy(pred, label, reduction='none')
        p_t = label * pred + (1 - label) * (1 - pred)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.ap > 0:
            alpha_factor = label * self.ap + (1 - label) * (1 - self.ap)
            loss *= alpha_factor
        return loss.mean()

class MakeLabel(nn.Module):
    def __init__(self):
        super(MakeLabel, self).__init__()

    def forward(self, anchor, label, cls_label=0, labelnum=1):
        # 计算 IoU
        iou = call_iou(anchor, label)
        # 初始化 label 为 0 的张量
        labels = torch.zeros((anchor.shape[0], anchor.shape[1], labelnum), device=anchor.device)
        
        # 找出 IoU >= 0.5 的位置
        indices = torch.nonzero(iou >= 0.5, as_tuple=True)  # 获取满足条件的索引
        # 使用这些索引设置相应位置的 labels
        labels[indices[0], indices[1], cls_label] = 1

        return labels




if __name__ == '__main__':
    from anchors import Anchors
    a = Anchors().cuda()
    seed = torch.rand(2,256,224,224).cuda()
    anchor=a(seed).repeat(2,1,1)
    label = torch.rand(2,4).cuda()
    ml=MakeLabel().cuda()
    label_=ml(anchor,label,cls_label=0,labelnum=1)
    for i in label:
        if i[0] == 1:
            print(6)
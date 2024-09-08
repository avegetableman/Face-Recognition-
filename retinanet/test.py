import torch
from anchor import Anchors
anchor = Anchors()
seed = torch.rand(1,256,224,224)
ansewr = torch.cat([torch.tensor(anchor(seed,x)) for x in range(5)],dim=1)
def make_regress_label(anchornum,label):
    target = []
    for epoch,anchor in enumerate(label):
        tmp=[]
        for i in range(anchornum):
            tmp.append(anchor.detach().numpy())
        target.append(tmp)
    return torch.tensor(target)
print(make_regress_label(ansewr.shape[1],torch.rand(1,4)).shape)
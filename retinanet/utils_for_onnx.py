import numpy as np

def iou(box0,box1):
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

class Anchors():
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)
        return all_anchors.astype(np.float32)

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

class transform_box():
    def __init__(self,anchor,size=(224,224)):
        super(transform_box, self).__init__()
        self.anchor = anchor()
        self.seed = self.anchor.forward(np.random.rand(1,3,size[0],size[1]).astype(np.float32))
    def forward(self,answer,th=0.4): #only one batch and cpu
        zb_answer,cls_answer = answer
        cls_answer = cls_answer[0]
        zb_answer = zb_answer[0]
        index = cls_answer > th
        index2 = np.repeat(index, 4, axis=1)
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

class NMS():
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
    a=Anchors()
    x=np.random.rand(1,3,224,224).astype(np.float32)
    y=a.forward(x)
    print(len(y[0]))
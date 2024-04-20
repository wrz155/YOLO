import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def yolo_decode(output, num_classes, anchors, num_anchors, scale_x_y):
    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    n_ch = 4+1+num_classes
    A = num_anchors
    B = output.size(0)
    H = output.size(2)
    W = output.size(3)

    output = output.view(B, A, n_ch, H, W).permute(0,1,3,4,2).contiguous()
    bx, by = output[..., 0], output[..., 1]
    bw, bh = output[..., 2], output[..., 3]

    det_confs = output[..., 4]
    cls_confs = output[..., 5:]

    bx = torch.sigmoid(bx)
    by = torch.sigmoid(by)
    bw = torch.exp(bw)*scale_x_y - 0.5*(scale_x_y-1)
    bh = torch.exp(bh)*scale_x_y - 0.5*(scale_x_y-1)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    grid_x = torch.arange(W, dtype=torch.float).repeat(1, 3, W, 1).to(device)
    grid_y = torch.arange(H, dtype=torch.float).repeat(1, 3, H, 1).permute(0, 1, 3, 2).to(device)
    bx += grid_x
    by += grid_y

    for i in range(num_anchors):
        bw[:, i, :, :] *= anchors[i*2]
        bh[:, i, :, :] *= anchors[i*2+1]

    bx = (bx / W).unsqueeze(-1)
    by = (by / H).unsqueeze(-1)
    bw = (bw / W).unsqueeze(-1)
    bh = (bh / H).unsqueeze(-1)

    #boxes = torch.cat((x1,y1,x2,y2), dim=-1).reshape(B, A*H*W, 4).view(B, A*H*W, 1, 4)
    boxes = torch.cat((bx, by, bw, bh), dim=-1).reshape(B, A * H * W, 4)
    det_confs = det_confs.unsqueeze(-1).reshape(B, A*H*W, 1)
    cls_confs =cls_confs.reshape(B, A*H*W, num_classes)
    # confs = (det_confs.unsqueeze(-1)*cls_confs).reshape(B, A*H*W, num_classes)
    outputs = torch.cat([boxes, det_confs, cls_confs], dim=-1)


    #return boxes, confs
    return outputs


class YoloLayer(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, img_size, anchor_masks=[], num_classes=80, anchors=[], num_anchors=9, scale_x_y=1):
        super(YoloLayer, self).__init__()
        #[6,7,8]
        self.anchor_masks = anchor_masks
        #类别
        self.num_classes = num_classes
        #
        if type(anchors) == np.ndarray:
            self.anchors = anchors.tolist()
        else:
            self.anchors = anchors

        print(self.anchors)
        print(type(self.anchors))

        self.num_anchors = num_anchors
        self.anchor_step = len(self.anchors) // num_anchors
        print(self.anchor_step)
        self.scale_x_y = scale_x_y

        self.feature_length = [img_size[0]//8,img_size[0]//16,img_size[0]//32]
        self.img_size = img_size

    def forward(self, output):
        if self.training:
            return output

        in_w = output.size(3)
        anchor_index = self.anchor_masks[self.feature_length.index(in_w)]
        stride_w = self.img_size[0] / in_w
        masked_anchors = []
        for m in anchor_index:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        self.masked_anchors = [anchor / stride_w for anchor in masked_anchors]

        data = yolo_decode(output, self.num_classes, self.masked_anchors, len(anchor_index),scale_x_y=self.scale_x_y)
        return data
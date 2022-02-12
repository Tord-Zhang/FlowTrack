import sys

import torchvision

sys.path.append('core')

import argparse
import glob
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

DEVICE = 'cuda'

import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
import os


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--model',default="/home/zhimahu/jjy/FlowTrain/raft_pretained/raft-things.pth")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()

        # self.backbone = torch.nn.DataParallel(RAFT(args))
        # self.backbone.load_state_dict(torch.load(args.model))
        # self.backbone = self.backbone.module
        self.backbone = RAFT(args)
        ckpt = torch.load(args.model)
        new_weight = {}
        for key, val in ckpt.items():
            new_weight[key[7:]] = val
        self.backbone.load_state_dict(new_weight)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = torchvision.ops.RoIPool(output_size=[17,17],spatial_scale=1.0)
        self.fc = nn.Linear(289*2, 4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def normalize(self, image):
        mean = torch.as_tensor([0.485, 0.456, 0.406],
                               dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225],
                              dtype=image.dtype, device=image.device)
        image = (image - mean[None, :, None, None]) / std[None, :, None, None]
        return image


    def forward(self, x_pre, x, xpre_box):
        x_pre = self.normalize(x_pre)
        x = self.normalize(x)

        flow_low, flow_up = self.backbone(x_pre, x, iters=20, test_mode=True)

        roi = torch.cat((torch.zeros((xpre_box.shape[0], 1)).float().to(
                self.device, non_blocking=torch.cuda.is_available()), xpre_box), 1)

        flow = self.head(flow_up.to(torch.float32), roi)
        flow = flow.view(-1,2*17*17)
        flow = flow.to(torch.float32)

        x_box = self.fc(flow)
        return x_box




def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()



if __name__ == '__main__':
    input0 = torch.zeros(4, 3, 384, 384)
    input1 = torch.zeros(4, 3, 384, 384)
    bbox1 = torch.zeros(4, 4)
    bbox2 = torch.zeros(4, 4)

    model = Net()

    out = model(input0, input1, bbox1)

    print(out.shape)


import os
import sys
from collections import namedtuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numbers
from model import Net
import torchvision
from dataset import FlowTrackDataset
from dataset_utils import Compose, RandomCrop, Resize, ToTensor

def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def calc_relative_motion_loss(pred, box_pre, box_current):
    ut_pred = box_pre[:, 0] + box_pre[:, 2] * pred[:, 0]
    vt_pred = box_pre[:, 1] + box_pre[:, 3] * pred[:, 1]
    wt_pred = torch.exp(pred[:, 2]) * box_pre[:, 2]
    ht_pred = torch.exp(pred[:, 3]) * box_pre[:, 3]

    abs_pos_pred = torch.stack((ut_pred, vt_pred, wt_pred, ht_pred)).permute(1, 0)
    print(box_current.shape, abs_pos_pred.shape)
    return F.smooth_l1_loss(abs_pos_pred, box_current, beta=4.0)



class TrackerSiamFC():

    def __init__(self, dataset_path, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__()
        self.cfg = self.parse_args(**kwargs)
        self.dataset_path = dataset_path

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net()
        init_weights(self.net)

        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        # self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
        self.lr_scheduler = MultiStepLR(self.optimizer, [20], 0.1)


    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0,
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        z_box = batch[2].to(self.device, non_blocking=self.cuda)
        x_box = batch[3].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            predict_box = self.net(z, x, z_box)
            predict_box = predict_box.to(torch.float32)
            x_box = x_box.to(torch.float32)


            # calculate loss
            # loss = self.criterion(predict_box, x_box)
            loss = calc_relative_motion_loss(predict_box, z_box, x_box)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ###############if you need to adjust the input size, specify it here####################
        transforms = Compose(
            [
                RandomCrop(),
                Resize(length=384, multiple=1),
                ToTensor()
            ]
        )
        #######################################################################################
        dataset = FlowTrackDataset(self.dataset_path, transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
                exit()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'flow_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)


if __name__ == '__main__':
    root_dir = os.path.expanduser('data/ILSVRC2015_VID/ILSVRC2015')

    tracker = TrackerSiamFC(root_dir)
    tracker.train_over()

# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int
from udl_vis.Basis.variance_sacling_initializer import variance_scaling_initializer
# from UDL.pansharpening.models import PanSharpeningModel
# from models.base_model import PanSharpeningModel

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                print("nn.Conv2D is initialized by variance_scaling_initializer")
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


# -----------------------------------------------------
class PanNet(nn.Module):
    def __init__(self, spectral_num, criterion, channel=32, reg=True):
        super(PanNet, self).__init__()
        self.criterion = criterion
        self.reg = reg

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        self.apply(init_weights)
        # init_weights(self.backbone, self.deconv, self.conv1, self.conv3)  # state initialization, important!

    def forward(self, x, y):# x= hp of ms; y = hp of pan

        output_deconv = self.deconv(x)
        input = torch.cat([output_deconv, y], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!

        output = self.conv3(rs)  # Bsx8x64x64
        return output

    def train_step(self, data, *args, **kwargs):
        return lms + self(data['ms_hp'], data['pan_hp'])

    def val_step(self, data, *args, **kwargs):
        return lms + self(data['ms_hp'], data['pan_hp'])


# from UDL.pansharpening.models import PanSharpeningModel
from pancollection.models.base_model import PanSharpeningModel
from udl_vis.Basis.criterion_metrics import SetCriterion

class build_pannet(PanSharpeningModel, name='PanNet'):
    def __call__(self, cfg):

        if not all(['hp' in name for name in list(cfg.dataset.values())]):
            raise ValueError(f"{cfg.dataset} is wrong for PanNet, you need high-pass filter dataset.")

        # important for Pansharpening models, which are from tensorflow code
        self.reg = cfg.reg

        scheduler = None

        if any(["wv" in v for v in cfg.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4
        loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = PanNet(spectral_num, criterion).cuda()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)   ## optimizer 1: Adam

        return model, criterion, optimizer, scheduler

# ----------------- End-Main-Part ------------------------------------

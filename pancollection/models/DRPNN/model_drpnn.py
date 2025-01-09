# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, Ran Ran
# @reference:
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int

import torch
import torch.nn as nn
import math
from udl_vis.Basis.variance_sacling_initializer import variance_scaling_initializer
# from UDL.pansharpening.models import PanSharpeningModel

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                print("nn.Conv2D is initialized by variance_scaling_initializer")
                variance_scaling_initializer(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if 'conv' in k and 'weight' in k:
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))

        loss = criterion + sum(regularizations)
        return loss


class Repeatblock(nn.Module):
    def __init__(self):
        super(Repeatblock, self).__init__()

        channel = 32  # input_channel =
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs = self.relu(self.conv2(x))

        return rs

class DRPNN(nn.Module):
    def __init__(self, spectral_num, criterion, channel=64):
        super(DRPNN, self).__init__()

        channel = 32  # input_channel =
        # spectral_num = 8
        self.criterion = criterion
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num+1, out_channels=channel, kernel_size=7, stride=1, padding=3,
                                  bias=True)

        # self.repeat1 = Repeatblock()
        # self.repeat2 = Repeatblock()
        # self.repeat3 = Repeatblock()
        # self.repeat4 = Repeatblock()
        # self.repeat5 = Repeatblock()
        # self.repeat6 = Repeatblock()
        # self.repeat7 = Repeatblock()
        # self.repeat8 = Repeatblock()
        # self.repeat9 = Repeatblock()

        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num+1, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.conv3 = nn.Conv2d(in_channels=spectral_num+1, out_channels=spectral_num, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
        )
        self.apply(init_weights)
        # init_weights(self.backbone, self.conv1, self.conv3, self.conv4)   # state initialization, important!

    def forward(self, x, y):  # x= lms; y = pan

        input = torch.cat([x, y], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx64x64x64

        rs = self.backbone(rs)  # backbone!  Bsx64x64x64

        out_res = self.conv2(rs)  # Bsx9x64x64
        output1 = torch.add(input, out_res)  # Bsx9x64x64
        output  = self.conv3(output1)  # Bsx8x64x64

        return output

    def train_step(self, data, *args, **kwargs):
        return self(data['lms'], data['pan'])

    def val_step(self, data, *args, **kwargs):
        return self(data['lms'], data['pan'])

# ----------------- End-Main-Part ------------------------------------

# from UDL.pansharpening.models import PanSharpeningModel
from pancollection.models.base_model import PanSharpeningModel
from udl_vis.Basis.criterion_metrics import SetCriterion

class build_drpnn(PanSharpeningModel, name='DRPNN'):
    def __call__(self, cfg):

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
        model = DRPNN(spectral_num, criterion).cuda()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)  ## optimizer 1: Adam
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1500,
                                                       gamma=0.5)  # lr = lr* gamma for each step_size = 180

        return model, criterion, optimizer, scheduler




if __name__ == '__main__':
    lms = torch.randn([1, 8, 64, 64])
    pan = torch.randn([1, 8, 64, 64])
    model = DRPNN(8, None)
    x = model(lms, pan)
    print(x.shape)
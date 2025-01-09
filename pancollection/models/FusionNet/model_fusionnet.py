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
from udl_vis.Basis.module import PatchMergeModule


class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if "conv" in k and "weight" in k:
                # print(k)
                penality = weight_decay * ((v.data**2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = criterion + sum(regularizations)
        return loss


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                print("initial nn.Conv2d with var_scale_new: ", m)
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv21 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


class FusionNet(PatchMergeModule):
    def __init__(self, spectral_num, channel=32):
        super(FusionNet, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num
        # self.criterion = criterion

        self.conv1 = nn.Conv2d(
            in_channels=spectral_num,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.conv3 = nn.Conv2d(
            in_channels=channel,
            out_channels=spectral_num,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

        # self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
        #     *[Resblock() for _ in range(4)]
        # )
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.backbone = nn.Sequential(self.res1, self.res2, self.res3, self.res4)

        # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
        # self.apply(init_weights)

    def forward(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat.clone(), x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output

    def train_step(self, data, *args, **kwargs):
        sr =  data["lms"] + self.forward(data["lms"], data["pan"])
        loss = self.criterion(sr, data["gt"])
        return loss

    def val_step(self, data, *args, **kwargs):
        return data["lms"] + self.forward(data["lms"], data["pan"])


from torch import optim
from pancollection.models.base_model import PanSharpeningModel
from udl_vis.Basis.criterion_metrics import SetCriterion


class build_fusionnet(PanSharpeningModel, name="FusionNet"):
    def __call__(self, args):
        scheduler = None
        if any(["wv" in v for v in args.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4

        loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {"loss": 1}
        losses = {"loss": loss}
        criterion = SetCriterion(losses, weight_dict)
        model = FusionNet(spectral_num).cuda()
        model.criterion = criterion
        optimizer = optim.Adam(
            model.parameters(), lr=float(args.lr), weight_decay=0
        )  ## optimizer 1: Adam

        return model, criterion, optimizer, scheduler

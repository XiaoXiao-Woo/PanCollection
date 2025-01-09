# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from udl_vis.Basis.variance_sacling_initializer import variance_scaling_initializer
# from UDL.pansharpening.models import PanSharpeningModel
# from models.base_model import PanSharpeningModel

class PNN(nn.Module):
    def __init__(self, spectral_num, criterion, channel=64):
        super(PNN, self).__init__()

        self.criterion = criterion

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=9, stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        # init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):  # x = cat(lms,pan)
        input1 = x  # Bsx9x64x64

        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        output = self.conv3(rs)

        return output

    def train_step(self, data, *args, **kwargs):
        blk = self.blk
        gt = data['gt'][:, :, blk:-blk, blk:-blk]
        lms = torch.cat([data['lms'], data['pan']], dim=1)
        sr = self(lms)
        return sr

    def val_step(self, data, *args, **kwargs):
        blk = self.blk
        test_I_in1 = torch.cat([data['lms'], data['pan']], dim=1)
        test_I_in1 = F.pad(test_I_in1, (blk, blk, blk, blk), mode='replicate')
        sr = self(test_I_in1)
        return sr

    @classmethod
    def set_blk(cls, blk):
        cls.blk = blk

from pancollection.models.base_model import PanSharpeningModel
from udl_vis.Basis.criterion_metrics import SetCriterion
class build_pnn(PanSharpeningModel, name='PNN'):
    def __call__(self, cfg):

        # important for Pansharpening models, which are from tensorflow code
        self.reg = cfg.reg


        scheduler = None

        if any(["wv" in v for v in cfg.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4
        lr = 0.0001 * 17 * 17 * spectral_num
        cfg.lr = lr
        print(f"PNN adopted another lr: {lr} in \"build_pnn in pnn_main.py\" ")


        loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = PNN(spectral_num, criterion).cuda()
        target_layerParam = list(map(id, model.conv3.parameters()))
        base_layerParam = filter(lambda p: id(p) not in target_layerParam, model.parameters())

        training_parameters = [{'params': model.conv3.parameters(), 'lr': lr / 10},
                               {'params': base_layerParam}]

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  ## optimizer 2: SGD

        net_scope = 0
        for name, layer in model.named_parameters():
            if 'conv' in name and 'bias' not in name:
                net_scope += layer.shape[-1] - 1

        net_scope = np.sum(net_scope) + 1
        blk = net_scope // 2  # 8
        model.set_blk(blk)

        return model, criterion, optimizer, scheduler

# ----------------- End-Main-Part ------------------------------------

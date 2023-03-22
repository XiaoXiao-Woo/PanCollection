# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from UDL.Basis.criterion_metrics import SetCriterion
from UDL.pansharpening.models import PanSharpeningModel

BN_MOMENTUM = 0.01

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ChannelFuseBlock(nn.Module):
    expansion = 1

    def __init__(self, spectral_num, inplanes, planes, num_res=1, stride=1, downsample=None):
        super(ChannelFuseBlock, self).__init__()
        self.conv_up = conv3x3(spectral_num, planes, stride)
        self.epsilon = 1e-4
        self.rs_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()

        self.res_block = nn.ModuleList([])
        num_res = 1
        for i in range(num_res):
            self.res_block.append(BasicBlock(planes, planes, stride, downsample))

    def forward(self, inputs):
        # Pan + Ms 1,8,H,W + 1,8,H,W
        x, y = inputs[0], inputs[1]

        rs_w = self.relu(self.rs_w)
        weight = rs_w / (torch.sum(rs_w, dim=0) + self.epsilon)

        # y = self.conv_up(torch.cat([x, y], dim=1))
        y = self.conv_up(y)

        out = weight[0] * x + weight[1] * y
        out = self.relu(out)
        for res_conv in self.res_block:
            out_rs = res_conv(out)

        if len(self.res_block) != 1:
            out_rs = out + out_rs

        return out_rs


class HighResolutionModule(nn.Module):
    '''
    高低分支交叉 前后branches数
    '''

    def __init__(self, spectral_num, num_branches, blocks, num_blocks, num_inchannels, num_channels_pre_layer,
                 num_channels_cur_layer,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.epsilon = 1e-4
        self.num_inchannels = num_inchannels
        self.num_channels_pre_layer = num_channels_pre_layer
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        if num_branches == 2:
            self.transition_layers = self._our_make_transition_layer(spectral_num, num_channels_pre_layer, num_channels_cur_layer)

        self.relu = nn.ReLU(inplace=True)

        self.fcc_w = nn.Parameter(torch.ones(num_branches + 1, num_branches + 1, dtype=torch.float32),
                                  requires_grad=True)
        self.fcc_relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))

            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))

            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))

            raise ValueError(error_msg)

    # 构建具体的某一分支网络
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        layers = []
        # 加深卷积层，仅第一层牵涉到下采样
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    # 用于构建分支
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)
    def _our_make_transition_layer(self, spectral_num, num_channels_pre_layer, num_channels_cur_layer):

        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                transfpn = []
                if i + 1 - num_branches_pre > 0:
                    transfpn.append(nn.Sequential(
                        TransitionFPN(spectral_num, len(num_channels_cur_layer), 0, 0,
                                      kernel_size=3, stride=2, padding=1)))

                transition_layers.append(nn.Sequential(*transfpn))
        # 获得同一个输入，不支持不同输入
        return nn.ModuleList(transition_layers)


    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1,
                                          bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1,
                                          bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, inputs):

        x, y = inputs[0], inputs[1]

        num_branches = self.num_branches
        fcc_w = self.fcc_relu(self.fcc_w)
        weight = fcc_w / (torch.sum(fcc_w, dim=0) + self.epsilon)

        if num_branches == 1:
            return [self.branches[0](x[0])]


        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        #################################################################################
        if num_branches == 2:
            x_new_branch = self.transition_layers[num_branches]((*x, y))

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                weight_m = weight[i][0] if j == 1 else 1
                if i == j:
                    y = weight_m * y + weight[i][j] * x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = weight_m * y + weight[i][j] * F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
                else:
                    y = weight_m * y + weight[i][j] * self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        if num_branches == 2:
            for i in range(num_branches):
                if self.transition_layers[i] is not None:
                    if i < num_branches - 1:
                        x_fuse[i] = self.transition_layers[i](x_fuse[i])
            x_fuse.append(x_new_branch)

        return x_fuse


class TransitionFPN(nn.Module):
    def __init__(self, spectral_num, num_branches_after_trans, inchannels=0, outchannels=0, kernel_size=3, stride=1, padding=1):
        super(TransitionFPN, self).__init__()
        self.num_branches = num_branches_after_trans  # num_branches_cur=2,3
        if self.num_branches == 2:
            self.b0_in_down = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                                            nn.ReLU())
            self.cfs_layers = nn.Sequential(ChannelFuseBlock(spectral_num, 64, 64))
        if self.num_branches == 3:
            self.epsilon = 1e-4
            self.b0_in_down = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                                            nn.ReLU())
            self.b1_down = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                         nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
                                         nn.ReLU())

            self.b1_in_down = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
                                            nn.ReLU())

            self.b2_conv_out = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                             nn.BatchNorm2d(128, momentum=BN_MOMENTUM))  # no relu
            self.cfs_layers = nn.Sequential(ChannelFuseBlock(spectral_num, 128, 128))

            self.rs_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()


    def forward(self, x):
        '''
        :param x: 32x64x64 | 32x64x64 64x32x32 | 32x64x64 64x32x32 128x16x16
        :param y: 8x64x64 | 8x32x32 | 8x16x16
        :return:
        '''
        # x = x[0]
        # out = self.conv3x3s(x)
        num_branches = self.num_branches
        # out = x
        #
        if num_branches == 2:
            b0_in, b1_in = x[0], x[1]  # ms = x[1]
            out = self.cfs_layers((self.b0_in_down(b0_in), b1_in))  # 做步进 等价于maxpool + conv
        if num_branches == 3:
            rs_w = self.relu(self.rs_w)
            weight = rs_w / (torch.sum(rs_w, dim=0) + self.epsilon)

            b0_in, b1_in, b2_in = x[0], x[1], x[2]  # ms = x[1]
            out = b1_in + self.b0_in_down(b0_in)  # 做步进 等价于maxpool + conv
            out = weight[0] * self.b1_in_down(b1_in) + weight[1] * self.cfs_layers((self.b1_down(out), b2_in))

        return out


class DCFNet(nn.Module):
    def __init__(self, spectral_num, criterion, mode="SUM"):
        super(DCFNet, self).__init__()
        self.first_fuse = mode
        if mode == "SUM":
            print("repeated sum head")
            init_channel = spectral_num
        elif mode == "C":
            print("concat head")
            init_channel = spectral_num + 1
        else:
            assert False, print("fisrt_fuse error")
        self.criterion = criterion
        self.blocks_list = [4, [4, 4], [4, 4, 4]]
        self.channels_list = [64, [32, 64], [32, 64, 128]]
        NUM_MODULES = 1  # HighResolutionModule cell repeat
        ################################################################################################
        # stem net
        self.conv1 = nn.Conv2d(init_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        ################################################################################################
        # stage 1
        self.layer1 = self._make_layer(block=Bottleneck, inplanes=64, planes=self.channels_list[0],
                                       blocks=self.blocks_list[0])
        stage1_out_channel = Bottleneck.expansion * self.channels_list[0]

        ################################################################################################
        self.Stage2_NUM_BRANCHES = 2

        num_channels = [
            self.channels_list[1][i] * BasicBlock.expansion for i in range(len(self.channels_list[1]))]

        self.transition1 = self._our_make_transition_layer(
            spectral_num,
            [stage1_out_channel], num_channels)
        transition_num_channels = [
            self.channels_list[2][i] * BasicBlock.expansion for i in range(len(self.channels_list[2]))]
        self.stage2, pre_stage_channels = self._make_stage(
            spectral_num=spectral_num,
            num_modules=NUM_MODULES, num_branches=self.Stage2_NUM_BRANCHES,
            num_blocks=self.blocks_list[1], num_inchannels=num_channels,
            num_channels_pre_layer=self.channels_list[1], num_channels_cur_layer=transition_num_channels,
            num_channels=self.channels_list[1], block=BasicBlock)

        ################################################################################################
        # 三个分支分别卷积
        self.Stage3_NUM_BRANCHES = 3
        num_channels = [
            self.channels_list[2][i] * BasicBlock.expansion for i in range(len(self.channels_list[2]))]

        self.stage3, pre_stage_channels = self._make_stage(
            spectral_num=spectral_num,
            num_modules=NUM_MODULES, num_branches=self.Stage3_NUM_BRANCHES,
            num_blocks=self.blocks_list[2], num_inchannels=num_channels,
            num_channels_pre_layer=pre_stage_channels, num_channels_cur_layer=num_channels,
            num_channels=self.channels_list[2], block=BasicBlock)

        ################################################################################################
        # 保主分支，进行最终分支融合输出
        last_inp_channels = np.int(np.sum(pre_stage_channels))  # ms_channels
        FINAL_CONV_KERNEL = 1
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=spectral_num,
                      kernel_size=FINAL_CONV_KERNEL, stride=1, padding=1 if FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_ms_fuse(self, num_branches, block, num_inchannels, num_channels, num_res_blocks, stride=1):
        branch_index = num_branches - 1
        downsample = None
        if stride != 1 or \
                num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM)
            )
        if torch.cuda.is_available():
            cfs_layers = nn.Sequential(ChannelFuseBlock(num_inchannels[branch_index], num_channels[branch_index],
                                                        num_res=num_res_blocks, stride=stride, downsample=downsample))
        else:
            cfs_layers = nn.Sequential(ChannelFuseBlock(num_inchannels[branch_index], num_channels[branch_index],
                                                        num_res=num_res_blocks, stride=stride, downsample=downsample))
        return cfs_layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        # 产生Bottleneck
        # stem主分支进行通道扩展时(64->256),下采样/2
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    # 为每个分支进行构造卷积模块
    def _make_stage(self, spectral_num, num_modules, num_branches, num_blocks, num_inchannels,
                    num_channels_pre_layer, num_channels_cur_layer,
                    num_channels, block, fuse_method="SUM", multi_scale_output=True):

        modules = []  # HRNet的多尺度目标检测
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(spectral_num,
                                     num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels_pre_layer,
                                     num_channels_cur_layer,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    # 全连接步进卷积,产生多分支
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        # BatchNorm2d(
                        #     num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(  # TODO 除了-8外怎么平衡
                            inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _our_make_transition_layer(self, spectral_num, num_channels_pre_layer, num_channels_cur_layer):

        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                # conv3x3s = []
                transfpn = []
                if i + 1 - num_branches_pre > 0:
                    transfpn.append(nn.Sequential(
                        TransitionFPN(spectral_num, len(num_channels_cur_layer), 0, 0,
                                      kernel_size=3, stride=2, padding=1)))

                transition_layers.append(nn.Sequential(*transfpn))
        # 获得同一个输入，不支持不同输入
        return nn.ModuleList(transition_layers)

    def forward(self, x, ly, my, sy):
        '''
        TODO: ly三层卷后级联x再做transition_layer
        Input Image kind: Pan or ms
        x: high resolution Image is inputed in Network, shape:[N, C, 64, 64]
        y: low resolution image is used to fine-tune PAN image to produce lms,
                  shape:[N, C, base_rsl=16, base_rsl=16],
                        [N, C, base_rsl*scale_up, base_rsl*scale_up],
                        [N, C, base_rsl*scale_up, base_rsl*scale_upper]
        :return: higher resolution image x, shape:[N, C, 64, 64]
        '''
        if self.first_fuse == "SUM":
            x = x.repeat(1, ly.shape[1], 1, 1)
            x = x + ly
        if self.first_fuse == "C":
            x = torch.cat([x, ly], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 64->256,64x64
        x = self.layer1(x)

        # 扩展网络主干
        x_list = []
        for i in range(self.Stage2_NUM_BRANCHES):
            if self.transition1[i] is not None:
                if i < self.Stage2_NUM_BRANCHES - 1:
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(self.transition1[i]((x, my)))
            else:
                x_list.append(x)

        x_list = self.stage2((x_list, sy))
        x = self.stage3((x_list, None))
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2], 1)

        x = self.last_layer(x)

        return x

    def init_weights(self, pretrained='', ):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                print('=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def train_step(self, data):

        lms = data['lms'].cuda()
        ms = data['ms'].cuda()
        mms = torch.nn.functional.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
                                                   mode="bilinear", align_corners=True)
        pan = data['pan'].cuda()
        gt = data['gt'].cuda()
        sr = self(pan, lms, mms, ms)
        loss, log_vars = self.criterion(sr, gt)

        outputs = dict(sr=sr, loss=loss,
                       log_vars=log_vars,
                       num_samples=len(data['pan'].data))

        return outputs#sr, loss

    def val_step(self, data, *args, **kwargs):
        lms = data['lms'].cuda().float()
        ms = data['ms'].cuda().float()
        # if hasattr(data, 'mms') is not None:
        mms = data['mms'].cuda().float()
        # else:
        #     mms = torch.nn.functional.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2),
        #                                                mode="bilinear", align_corners=True)
        pan = data['pan'].cuda().float()
        sr = self(pan, lms, mms, ms) + lms
        return sr, data['gt'].cuda().float()


class build_DCFNet(PanSharpeningModel, name='DCFNet'):
    def __call__(self, args):
        scheduler = None
        if any(["wv" in v for v in args.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4
        # spectral_num = 4
        loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = DCFNet(spectral_num, criterion, mode="C").cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)  ## optimizer 1: Adam

        return model, criterion, optimizer, scheduler

if __name__ == "__main__":
    from UDL.Basis.auxiliary import torchstat
    loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
    weight_dict = {'loss': 1}
    losses = {'loss': loss}
    criterion = SetCriterion(losses, weight_dict)
    model = DCFNet(8, criterion, mode="C").cuda()
    torchstat.stat(model, [[1, 1, 256, 256], [1, 8, 256, 256], [1, 8, 128, 128], [1, 8, 64, 64]])

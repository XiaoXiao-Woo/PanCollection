import math

import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
# from unet.datasets import *

def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(4, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def summaries(model, writer=None, grad=False, torchsummary=None):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))


class SSconv(nn.Module):

    def __init__(self, in_channel, up):
        super().__init__()
        self.conv_up1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * up * up, kernel_size=3,
                                 stride=1, padding=1, bias=True)
        self.up_size = up

    def mapping(self, x):
        B, C, H, W = x.shape
        C1, H1, W1 = C // (self.up_size * self.up_size), H * self.up_size, W * self.up_size
        x = x.reshape(B, C1, self.up_size, self.up_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C1, H1, W1)
        return x

    def forward(self, x):
        x = self.conv_up1(x)
        return self.mapping(x)


class Down(nn.Module):

    def __init__(self):
        super().__init__()
        self.max_pool_conv = nn.MaxPool2d(2)

    def forward(self, x):
        return self.max_pool_conv(x)

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class pandown(nn.Module):

    def __init__(self):
        super(pandown, self).__init__()
        self.conv_down1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2,
                                          stride=2, padding=0, bias=True)
        self.conv_down2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4,
                                          stride=4, padding=0, bias=True)

        init_weights(self.conv_down1, self.conv_down2)

    def forward(self, pan):
        return self.conv_down1(pan), self.conv_down2(pan)


class Unet(nn.Module):

    def __init__(self, pan_channels, ms_channels):
        super(Unet, self).__init__()
        spectral_num = ms_channels
        upscale1 = 2
        upscale2 = 4
        #   up_2
        self.conv_up1 = SSconv(spectral_num, upscale1)
        self.conv_up2 = SSconv(spectral_num, upscale2)

        self.pan_down = pandown()


        channel_input1 = pan_channels + ms_channels 
        channel_output1 = channel_input1 * 4    
        self.conv1 = Conv(channel_input1, channel_output1)
        channel_input2 = channel_output1 + ms_channels + 1  
        self.down1 = Down()
        channel_output2 = channel_output1 * 4    
        self.conv2 = Conv(channel_input2, channel_output2)
        channel_input3 = channel_output2 + ms_channels + 1  #
        self.down2 = Down()
        channel_output3 = channel_input3    
        self.conv3 = Conv(channel_input3, channel_output3)
        channel_input4 = channel_output3 + channel_output2  
        self.up1 = SSconv(channel_output3, 2)
        channel_output4 = 144    
        self.conv4 = Conv(channel_input4, channel_output4)
        channel_input5 = channel_output1 + channel_output4  
        self.up2 = SSconv(channel_output4, 2)
        channel_output5 = 36   
        self.conv5 = Conv(channel_input5, channel_output5)

        # self.conv_mid1 = Conv(channel_output1, channel_output1)
        # self.conv_mid2 = Conv(channel_output2, channel_output2)

        self.O_conv3 = OutConv(channel_output3, ms_channels)
        self.O_conv4 = OutConv(channel_output4, ms_channels)
        self.O_conv5 = OutConv(channel_output5, ms_channels)

        init_weights(self.conv_up1, self.conv_up2,
                     self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.down1, self.down2, self.up1, self.up2,
                     self.O_conv3, )
    def forward(self, ms, pan):
        dim = ms.dim() - 3
        panda1, panda2 = self.pan_down(pan)
        # print(type(panda2))
        # print(panda2.shape)
        # print(type(ms))
        # print(ms.shape)
        # ms1 = nn.functional.interpolate(ms, scale_factor=2, mode='nearest')
        # ms2 = nn.functional.interpolate(ms, scale_factor=4, mode='nearest')
        ms_up1 = self.conv_up1(ms)
        # print(ms_up1.shape)
        # input()
        ms1 = torch.cat((ms_up1, panda1), dim)

        ms2 = self.conv_up2(ms)

        ms = torch.cat((ms, panda2), dim)

        x1 = self.conv1(torch.cat((pan, ms2), dim))
        # print("x1:", x1.shape)
        x2 = self.down1(x1)
        # print("x2:", x2.shape)
        x2 = self.conv2(torch.cat((x2, ms1), dim))
        # print("x2:", x2.shape)
        x3 = self.down2(x2)
        # print("x3:", x3.shape)
        x3 = self.conv3(torch.cat((x3, ms), dim))
        # print("x3:", x3.shape)
        x4 = self.up1(x3)
        # print("x4:", x4.shape)
        # x4 = self.conv4(torch.cat((x4, self.conv_mid2(x2)), dim))
        x4 = self.conv4(torch.cat((x4, x2), dim))
        # print("x4:", x4.shape)
        x5 = self.up2(x4)
        # print("x5:", x5.shape)
        # x5 = self.conv5(torch.cat((x5, self.conv_mid1(x1)), dim))
        x5 = self.conv5(torch.cat((x5, x1), dim))
        # print("x5:", x5.shape)
        x3 = self.O_conv3(x3)
        x4 = self.O_conv4(x4)
        x5 = self.O_conv5(x5)

        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        return x3, x4,

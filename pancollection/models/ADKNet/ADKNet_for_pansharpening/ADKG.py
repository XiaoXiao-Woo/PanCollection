import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain


# Filter normalization
class Norm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type,
                 nonlinearity='leaky_relu', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'spectral')
        assert in_channels >= 1
        super(Norm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'spectral':
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
            x = x.reshape(b, -1)
            if self.runing_std:
                x = x * self.std[None, :]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :]
        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x

# Spatial branch
def create_spatial_branch(in_channels, kernel_size=3,
                               nonlinearity='leaky_relu',
                               stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                  kernel_size=kernel_size, stride=stride,
                  padding=padding, groups=in_channels),
        nn.Conv2d(in_channels=in_channels, out_channels=kernel_size ** 2,
                  kernel_size=1),
        nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2,
                  kernel_size=kernel_size, padding=padding),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2,
                  kernel_size=kernel_size, padding=padding),
        Norm(in_channels=1, kernel_size=kernel_size,
             filter_type='spatial', nonlinearity=nonlinearity)
    )

# Spectral branch
def create_spectral_branch(in_channels, kernel_size=3,
                    nonlinearity='leaky_relu',
                    se_ratio=0.05):
    assert se_ratio > 0
    mid_channels = int(in_channels * se_ratio)
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                  kernel_size=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=mid_channels, out_channels=in_channels * kernel_size ** 2,
                  kernel_size=1),
        Norm(in_channels=in_channels, kernel_size=kernel_size,
             filter_type='spectral', nonlinearity=nonlinearity,
             running_std=True)
    )

# ADKG
class ADKGenerator(nn.Module):
    def __init__(self, in_channels, kernel_size=3,
                 nonlinearity='leaky_relu', stride=1,
                 padding=1, se_ratio=0.05):
        super(ADKGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        self.spatial_branch = create_spatial_branch(in_channels=in_channels, kernel_size=kernel_size,
                                                    nonlinearity=nonlinearity, stride=stride,
                                                    padding=padding)
        self.spectral_branch = create_spectral_branch(in_channels=in_channels, kernel_size=kernel_size,
                                                      nonlinearity=nonlinearity, se_ratio=se_ratio)

    def forward(self, x, y):  # x stands for LR-MS features while y stands for PAN features
        b, c, h, w = x.shape
        k = self.kernel_size
        spatial_kernel = self.spatial_branch(y).permute(0, 2, 3, 1).reshape(b, 1, h, w, k, k)
        spectral_kernel = self.spectral_branch(x).reshape(b, c, 1, 1, k, k)
        self.adaptive_discriminative_kernel = torch.mul(spectral_kernel, spatial_kernel)
        output = self.AC4IF(x)
        return output

    def AC4IF(self, x):  # adaptive convolution for image fusion
        b, c, h, w = x.shape
        pad = self.padding
        k = self.kernel_size
        kernel = self.adaptive_discriminative_kernel
        x_pad = torch.zeros(b, c, h + 2 * pad, w + 2 * pad,
                            device='cuda:0')
        if pad > 0:
            x_pad[:, :, pad:-pad, pad:-pad] = x
        else:
            x_pad = x
        x_pad = F.unfold(x_pad, (k, k))
        x_pad = x_pad.reshape(b, c, k, k, h, w).permute(0, 1, 4, 5, 2, 3)

        return torch.sum(torch.mul(x_pad, kernel), [4, 5])

# Fast ADKG
from ddf import DDFFunction
ddf = DDFFunction.apply
class Fast_ADKGenerator(nn.Module):
    def __init__(self, in_channels, kernel_size=3,
                 nonlinearity='leaky_relu', stride=1,
                 padding=1, se_ratio=0.05, kernel_combine='mul'):
        super(Fast_ADKGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_combine = kernel_combine

        self.spatial_branch = create_spatial_branch(in_channels=in_channels, kernel_size=kernel_size,
                                                    nonlinearity=nonlinearity, stride=stride,
                                                    padding=padding)
        self.spectral_branch = create_spectral_branch(in_channels=in_channels, kernel_size=kernel_size,
                                                      nonlinearity=nonlinearity, se_ratio=se_ratio)

    def forward(self, x, y):  # x stands for LR-MS features while y stands for PAN features
        b, c, h, w = x.shape
        k = self.kernel_size
        spatial_kernel = self.spatial_branch(y).reshape(b, -1, h, w)
        spectral_kernel = self.spectral_branch(x).reshape(b, c, k, k)
        output = ddf(x, spectral_kernel, spatial_kernel, self.kernel_size, 1, 1,
                     self.kernel_combine).type_as(x)
        output = output.reshape(b, c, h, w)
        return output



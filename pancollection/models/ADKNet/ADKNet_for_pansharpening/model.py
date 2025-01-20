import torch.nn as nn
from ADKG import ADKGenerator
#from ADKG import Fast_ADKGenerator as ADKGenerator


class ADKLayer(nn.Module):
    def __init__(self, channel=60, kernel_size=3, nonlinearity='leaky_relu',
                 stride=1, se_ratio=0.05):
        super(ADKLayer, self).__init__()
        self.ADKGenerator = ADKGenerator(in_channels=channel, kernel_size=kernel_size,
                               nonlinearity=nonlinearity, stride=stride,
                               se_ratio=se_ratio)

    def forward(self, x, y):
        x = self.ADKGenerator(x, y)
        return x


class ADKNet(nn.Module):
    def __init__(self, criterion, spectral_num):
        super(ADKNet, self).__init__()

        self.criterion = criterion
        channel = 60
        # spectral_num = 8

        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num,
                                         kernel_size=8, stride=4, padding=2, bias=True)
        self.upsample = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num,
                                           kernel_size=8, stride=4, padding=2, bias=True)

        self.adk1 = ADKLayer()
        self.adk2 = ADKLayer()
        self.adk3 = ADKLayer()
        self.adk4 = ADKLayer()
        self.adk5 = ADKLayer()
        self.adk6 = ADKLayer()
        self.adk7 = ADKLayer()
        self.relu = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=5, padding=2)

    def forward(self, x, y):  # x stands for lrms, and y stands for pan

        skip = self.upsample(x)

        # forward propagation
        x = self.relu(self.deconv(x))
        x = self.relu(self.conv1(x))
        y = self.relu(self.conv3(y))

        # 7 ADK layers
        x = self.adk1(x, y)
        x = self.adk2(x, y)
        x = self.adk3(x, y)
        x = self.adk4(x, y)
        x = self.adk5(x, y)
        x = self.adk6(x, y)
        x = self.adk7(x, y)

        x = self.conv2(x)

        return x + skip


    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()
        sr = self(ms, pan)
        loss = self.criterion(sr, gt, *args, **kwargs)['loss']
        # outputs = loss
        # return loss
        log_vars.update(loss=loss.item())
        metrics = {'loss': loss, 'log_vars': log_vars}
        return metrics

    def val_step(self, data, *args, **kwargs):
        # gt, lms, ms, pan = data
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()
        sr = self(ms, pan)

        return sr, gt

import torch
from torch import optim
from udl_vis.Basis.criterion_metrics import SetCriterion
from pancollection.models import PanSharpeningModel

class build_ADKNet(PanSharpeningModel, name='ADKNet'):
    def __call__(self, args):
        scheduler = None
        if any(["wv" in v for v in args.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4


        loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = ADKNet(criterion, spectral_num)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.5)

        return model, criterion, optimizer, scheduler

if __name__ == '__main__':
    HSI = torch.randn([1, 8, 16, 16]).cuda()
    MSI = torch.randn([1, 1, 64, 64]).cuda()
    model = ADKNet(None, 8).cuda()

    print(model(HSI, MSI).shape)
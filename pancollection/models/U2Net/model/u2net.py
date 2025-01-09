import torch
import torch.nn as nn
from s2block import S2Block


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*16, 3, 1, 1, bias=False),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up1(x1)
        x = x1 + x2
        return self.relu(self.conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class U2Net(nn.Module):
    def __init__(self, dim, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super().__init__()
        ms_dim = 8
        pan_dim = 1

        self.relu = nn.LeakyReLU()
        self.upsample = Upsample(ms_dim)
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(ms_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(pan_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )
        
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)
        dim3 = dim1
        dim4 = dim0

        # layer 0
        self.s2block0 = S2Block(dim0, dim0//dim_head, dim_head, int(dim0*se_ratio_mlp))
        self.down0 = Down(dim0, dim1)
        self.resblock0 = ResBlock(dim0, int(se_ratio_rb*dim0), dim0)

        # layer 1
        self.s2block1 = S2Block(dim1, dim1//dim_head, dim_head, int(dim1*se_ratio_mlp))
        self.down1 = Down(dim1, dim2)
        self.resblock1 = ResBlock(dim1, int(se_ratio_rb*dim1), dim1)

        # layer 2
        self.s2block2 = S2Block(dim2, dim2//dim_head, dim_head, int(dim2*se_ratio_mlp))
        self.up0 = Up(dim2, dim3)
        self.resblock2 = ResBlock(dim2, int(se_ratio_rb*dim2), dim2)

        # layer 3
        self.s2block3 = S2Block(dim3, dim3//dim_head, dim_head, int(dim3*se_ratio_mlp))
        self.up1 = Up(dim3, dim4)
        self.resblock3 = ResBlock(dim3, int(se_ratio_rb*dim3), dim3)

        # layer 4
        self.s2block4 = S2Block(dim4, dim4//dim_head, dim_head, int(dim4*se_ratio_mlp))

    def forward(self, x, y):
        x = self.upsample(x)
        skip_c0 = x
        x = self.raise_ms_dim(x)
        y = self.raise_pan_dim(y)

        # layer 0
        x = self.s2block0(x, y)  # 32 64 64
        skip_c10 = x  # 32 64 64
        x = self.down0(x)  # 64 32 32
        y = self.resblock0(y)  # 32 64 64
        skip_c11 = y  # 32 64 64
        y = self.down0(y)  # 64 32 32

        # layer 1
        x = self.s2block1(x, y)  # 64 32 32
        skip_c20 = x
        x = self.down1(x)  # 128 16 16
        y = self.resblock1(y)  # 64 32 32
        skip_c21 = y  # 64 32 32
        y = self.down1(y)  # 128 16 16

        # layer 2
        x = self.s2block2(x, y)  # 128 16 16
        x = self.up0(x, skip_c20)  # 64 32 32
        y = self.resblock2(y)  # 128 16 16
        y = self.up0(y, skip_c21)  # 64 32 32

        # layer 3
        x = self.s2block3(x, y)  # 64 32 32
        x = self.up1(x, skip_c10)  # 32 64 64
        y = self.resblock3(y)  # 64 32 32
        y = self.up1(y, skip_c11)  # 32 64 64

        # layer 4
        x = self.s2block4(x, y)  # 32 64 64
        output = self.to_hrms(x) + skip_c0  # 8 64 64
        
        return output


if __name__ == '__main__':
    ms = torch.randn([1, 8, 16, 16]).cuda()
    lms = torch.randn([1, 8, 64, 64]).cuda()
    pan = torch.randn([1, 1, 64, 64]).cuda()
    # x = torch.cat([lms, pan], dim=1).cuda()

    model = U2Net(dim=32).cuda()
    print(model(ms, pan).shape)


    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    # analysis = FlopCountAnalysis(model, (ms, pan))
    # print(flop_count_table(analysis))
    # # | model                            | 0.642M                 | 3.001G     |


    from fvcore.nn import FlopCountAnalysis, flop_count_table
    module = S2Block(dim=32, heads=8, dim_head=4, mlp_dim=32*2).cuda()
    x = torch.randn([1, 32, 64, 64]).cuda()
    y = torch.randn([1, 32, 64, 64]).cuda()
    analysis = FlopCountAnalysis(module, (x, y))
    print(flop_count_table(analysis))
    # | layers.0            | 9.472K                 | 1.115G    |

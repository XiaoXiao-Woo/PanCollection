import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from model.fusformer import Fusformer
from model.nn import (
    SiLU,
    conv_nd,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        model_channels=128,
        norm_type="gn",
        dropout=0.0,
        dims=2,
        use_scale_shift_norm=False
    ):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()
        emb_channels = model_channels * 4     # model_channel * 4
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                # out_channels,
                2 * out_channels if use_scale_shift_norm else out_channels
            ),
        )
        
        self.out_layers = nn.Sequential(
            normalization(norm_type, out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, out_channels, out_channels, 3, padding=1)
            ),
        )
        
    def time_emb(self, h, emb):
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return h
    
    
    def forward(self, x, emb):         # 32 64 64
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)   # 32 64 64
        rs1 = self.time_emb(rs1, emb)
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


from patch_merge_module.import_module import PatchMergeModule
class SSNet(PatchMergeModule):
    def __init__(
        self,
        dim,        # 32
        model_channels=128,
        dim_head=16,
        ms_dim=8,
        pan_dim=1,
        se_ratio_mlp=0.5,
        se_ratio_rb=0.5,
        dropout=0,
        device='cpu',
        norm_type="bn",
        crop_batch_size=1,
        use_scale_shift_norm=False,
    ):
        super().__init__(True, crop_batch_size, [64, 64, 16], device=device)
        ms_dim = ms_dim  # qb:4, gf2,  wv3:8
        # lms_dim = ms_dim
        pan_dim = pan_dim
        self.model_channels = model_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.device = device
        
        self.relu = nn.LeakyReLU()
        self.upsample = Upsample(ms_dim)
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),    # in_channels, out_channels, kernel_size, stride,padding
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.conv_lms2x_t = nn.Conv2d(ms_dim*2, dim, 3, 1, 1)    # 16 32
        self.conv_pan2x_t = nn.Conv2d(ms_dim+pan_dim, dim, 3, 1, 1)    # 9 32
        self.conv_x_t = nn.Conv2d(ms_dim, dim, 3, 1, 1)  # 8 32
        
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)
        dim3 = dim1
        dim4 = dim0

        # layer 0
        self.fusformer0 = Fusformer(dim0, dim0//dim_head, dim_head, int(dim0*se_ratio_mlp))
        self.down0 = Down(dim0, dim1)
        self.resblock0 = ResBlock(dim0, int(se_ratio_rb*dim0), dim0,
                                  model_channels=self.model_channels, use_scale_shift_norm=use_scale_shift_norm) # 32 16 32 128
 
        # layer 1
        self.fusformer1 = Fusformer(dim1, dim1//dim_head, dim_head, int(dim1*se_ratio_mlp))
        self.down1 = Down(dim1, dim2)
        self.resblock1 = ResBlock(dim1, int(se_ratio_rb*dim1), dim1, use_scale_shift_norm=use_scale_shift_norm)

        # layer 2
        self.fusformer2 = Fusformer(dim2, dim2//dim_head, dim_head, int(dim2*se_ratio_mlp))
        self.up0 = Up(dim2, dim3)
        self.resblock2 = ResBlock(dim2, int(se_ratio_rb*dim2), dim2, use_scale_shift_norm=use_scale_shift_norm)

        # layer 3
        self.fusformer3 = Fusformer(dim3, dim3//dim_head, dim_head, int(dim3*se_ratio_mlp))
        self.up1 = Up(dim3, dim4)
        self.resblock3 = ResBlock(dim3, int(se_ratio_rb*dim3), dim3, use_scale_shift_norm=use_scale_shift_norm)

        # layer 4
        self.fusformer4 = Fusformer(dim4, dim4//dim_head, dim_head, int(dim4*se_ratio_mlp))
        # self.fusformer4 = Fusformer(dim4, dim4//8, 8, int(dim4*se_ratio_mlp))

        self.out_layers_pan = nn.Sequential(
            normalization(norm_type, dim),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(2, dim, dim, 3, padding=1)
            ),
        )
        
        self.out_layers_lms = nn.Sequential(
            normalization(norm_type, dim),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(2, dim, dim, 3, padding=1)
            ),
        )
        
        emb_channels = self.model_channels * 4
        self.emb_layers_pan = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                # 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                2 * dim if use_scale_shift_norm else dim
            ),
        )
        
        self.emb_layers_lms = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                # 2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                2 * dim if use_scale_shift_norm else dim,
            ),
        )
        
        
    def time_emb_pan(self, h, emb): # 24 32 64 64, 24 512
        emb_out = self.emb_layers_pan(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers_pan[0], self.out_layers_pan[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers_pan(h)
        return h
    
    def time_emb_lms(self, h, emb):
        emb_out = self.emb_layers_lms(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers_lms[0], self.out_layers_lms[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers_lms(h)
        return h
    
    def Fourier_filter(self, x, threshold, scale):
        import torch.fft as fft
        dtype = x.dtype
        x = x.type(torch.float32)
        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        
        B, C, H, W = x_freq.shape
        mask = torch.ones((B, C, H, W)).to(x.device)

        crow, ccol = H // 2, W //2
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
        x_freq = x_freq * mask

        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
        
        x_filtered = x_filtered.type(dtype)
        return x_filtered

    def forward_impl(self, lms, pan, ms, x_t, timesteps):    # x:lms , y:pan
        
        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        
        x = self.upsample(ms)
        skip_c0 = x
        pan = torch.cat([pan, x_t], dim=1)  # 9 64 64
        pan = self.conv_pan2x_t(pan)    # 32 64 64
        
        lms = torch.cat([lms, x_t], dim=1)  # 16 64 64
        lms = self.conv_lms2x_t(lms)    # 32 64 64
        
        y = self.time_emb_pan(pan, emb)   # 32 64 64
        x = self.time_emb_lms(lms, emb)   # 32 64 64
        
        # gf5
        b1=1.2
        b2=1.4
        b3=1.6
        s1=0.9
        s2=0.6
        s3=0.3
        # layer 0
        x[:,:16] = x[:,:16] * b1
        y0 = self.Fourier_filter(y, threshold=1, scale=s1)
        # y0 = y
        x = self.fusformer0(x, y0)  # 32 64 64
        
        skip_c10 = x  # 32 64 64
        x = self.down0(x)  # 64 32 32
        
        y = self.resblock0(y, emb)  # 32 64 64
        skip_c11 = y  # 32 64 64
        y = self.down0(y)  # 64 32 32

        # layer 1
        x[:,:32] = x[:,:32] * b2
        y1 = self.Fourier_filter(y, threshold=1, scale=s2)
        # y1 = y
        x = self.fusformer1(x, y1)  # 64 32 32
        skip_c20 = x
        x = self.down1(x)  # 128 16 16
        
        
        y = self.resblock1(y, emb)  # 64 32 32
        skip_c21 = y  # 64 32 32
        y = self.down1(y)  # 128 16 16

        # layer 2
        x[:,:64] = x[:,:64] * b3   # 128
        y2 = self.Fourier_filter(y, threshold=1, scale=s3)
        x = self.fusformer2(x, y2)  # 128 16 16
        x = self.up0(x, skip_c20)  # 64 32 32
        
        y = self.resblock2(y, emb)  # 128 16 16
        y = self.up0(y, skip_c21)  # 64 32 32

        # layer 3
        x[:,:32] = x[:,:32] * b2
        y3 = self.Fourier_filter(y, threshold=1, scale=s2)
        x = self.fusformer3(x, y3)  # 64 32 32
        x = self.up1(x, skip_c10)  # 32 64 64
        
        y = self.resblock3(y, emb)  # 64 32 32
        y = self.up1(y, skip_c11)  # 32 64 64

        # layer 4
        x[:,:16] = x[:,:16] * b1
        y4 = self.Fourier_filter(y, threshold=1, scale=s1)
        x = self.fusformer4(x, y4)  # 32 64 64


        output = self.to_hrms(x) + skip_c0  # 8 64 64

        
        return output

    def sample(self, 
               *args,
               **kwargs
               ):
        sample_fn = kwargs.pop('sample_fn')

        noise = kwargs['noise']
        shape = kwargs.pop('shape')
        device = args[0].device
        num_batch = args[0].shape[0]
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        img = img[:num_batch, ...]
        kwargs['noise'] = img
        
        self.indices = list(range(kwargs.pop('num_timesteps')))[::-1]
        if kwargs['progress'] and not hasattr(self, 'indices'):
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            self.indices = tqdm(self.indices)
        lms = args[0]

        for i in self.indices:
            t = torch.tensor([i] * shape[0], device=device)
            t = t[:num_batch, ...]          
            # out_patch[:, [4,2,0]]  
            out_patch = self.forward_impl(*args, x_t=img, timesteps=t)
            out_patch = sample_fn(out_patch, t, lms,
                                **kwargs)
            img = out_patch['sample']
            kwargs['noise'] = img

        return img.contiguous()

    
def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


def split_tensor(tensor, len):
    
    split_len = int(len/2)
    b = torch.split(tensor, split_len, dim=2)
    c1 = torch.split(b[0], split_len, dim=3)
    c2 = torch.split(b[1], split_len, dim=3)
    return [split_tensor(c1[0], split_len), split_tensor(c1[1], split_len),
            split_tensor(c2[0], split_len), split_tensor(c2[1], split_len)]

def concat_tensor(tensors):
    c1 = torch.cat((tensors[0], tensors[1]), 3)
    c2 = torch.cat((tensors[2], tensors[3]), 3)
    
    print(c1)
    print(c2)
    c = torch.cat((c1, c2), 2)
    print(c)
    return c
    

if __name__ == "__main__":
    import torch.nn.functional as F
    import einops
    import time
    from torch.cuda import max_memory_allocated
    device = 'cuda:1'
    model = SSNet(32).to(device)
    model.forward = model.forward_impl

    lms = torch.randn(1, 8, 256, 256).to(device)
    pan = torch.randn(1, 1, 256, 256).to(device)
    ms = torch.randn(1, 8, 64, 64).to(device)
    x_t = torch.randn(1, 8, 256, 256).to(device)
    t = torch.tensor([1.]).to(device)

    tic = time.time()
    model.eval()
    for _ in range(2000):
        sr = model.forward(lms, pan, ms, x_t, t)
    print(time.time() - tic)

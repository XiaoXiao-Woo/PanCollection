import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(x, y, **kwargs) + x
        else:
            return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        # self.temperature = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sa1 = nn.Linear(dim, inner_dim, bias=False)
        self.sa2 = nn.Linear(dim, inner_dim, bias=False)
        self.se1 = nn.Linear(dim, inner_dim, bias=False)
        self.se2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y, mask=None):
        b, n, _, h = *x.shape, self.heads
        y1 = rearrange(self.sa1(y), 'b n (h d) -> b h n d', h=h)
        y2 = rearrange(self.sa2(y), 'b n (h d) -> b h n d', h=h)
        x1 = rearrange(self.se1(x), 'b n (h d) -> b h n d', h=h)
        x2 = rearrange(self.se2(x), 'b n (h d) -> b h n d', h=h)
        sacm = (y1 @ y2.transpose(-2, -1)) * self.scale
        secm = (x1.transpose(-2, -1) @ x2) * self.scale / (n/self.dim_head)  # b h d d
        sacm = sacm.softmax(dim=-1)
        secm = secm.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', sacm, x1)
        out2 = torch.einsum('b h n i, b h i j -> b h n j', y1, secm)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out = out1 * out2
        out = self.to_out(out)
        return out


class S2Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, depth=1, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, MLP(dim, hidden_dim=mlp_dim, dropout=dropout)))]))

    def forward(self, x, y, mask=None):
        H = x.shape[2]
        x = rearrange(x, 'B C H W -> B (H W) C', H=H)
        y = rearrange(y, 'B C H W -> B (H W) C', H=H)
        for attn, ff in self.layers:
            x = attn(x, y, mask=mask)
            x = ff(x)
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x

import math

import torch
import torch.nn as nn
from torch.nn import init

# SE-ResNet from https://github.com/moskomule/senet.pytorch/blob/master/senet/
class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

# SE GenBlock
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), upsample=False, n_classes=0):
        super(GenBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        self.se1 = SELayer(out_channels)

        self.b1 = ConditionalBatchNorm2d(in_channels, n_classes)
        self.b2 = ConditionalBatchNorm2d(hidden_channels, n_classes)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def upsample_conv(self, x, conv):
        return conv(nn.UpsamplingNearest2d(scale_factor=2)(x))

    def residual(self, x, y):
        h = x
        h = self.b1(h, y)
        h = self.activation(h)
        h = self.upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y)
        h = self.activation(h)
        h = self.c2(h)
        h = self.se1(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y):
        return self.residual(x, y) + self.shortcut(x)

# 128*128
class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Generator, self).__init__()


        self.bottom_width = args.bottom_width #4
        self.activation = activation
        self.n_classes = args.n_classes
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.z_dim, (self.bottom_width ** 2) * self.ch) # 4*4*1024
        self.block2 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=self.n_classes) #8*8*1024
        self.block3 = GenBlock(self.ch, self.ch//2, activation=activation, upsample=True, n_classes=self.n_classes) #16*16*1024
        self.block4 = GenBlock(self.ch//2, self.ch//4, activation=activation, upsample=True, n_classes=self.n_classes) #32*32*256
        self.block5 = GenBlock(self.ch//4, self.ch//8, activation=activation, upsample=True, n_classes=self.n_classes) #64*64*128
        self.block6 = GenBlock(self.ch//8, self.ch//16, activation=activation, upsample=True, n_classes=self.n_classes) #128*128*64
        self.b7 = nn.BatchNorm2d(self.ch//16)
        # self.b7 = ConditionalBatchNorm2d(self.ch//16, self.n_classes)
        self.c7 = nn.Conv2d(self.ch//16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z, y):
        
        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.block6(h, y)
        h = self.b7(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c7(h))
 
        return h


"""Discriminator"""


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        # print(x.shape)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)

        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


# SE-DisBlock
class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        self.se1 = SELayer(out_channels)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)


    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = self.se1(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), dim_emb=128):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim #64
        self.n_clases = args.n_classes
        self.activation = activation
        
        self.block1 = OptimizedDisBlock(args, 3, self.ch) # 64
        self.block2 = DisBlock(args, self.ch, self.ch*2, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch*2, self.ch*4, activation=activation, downsample=True)
        self.embed = nn.Embedding(args.n_classes, dim_emb)
        self.block4 = DisBlock(args, self.ch*4+dim_emb, self.ch*8, activation=activation, downsample=True)
        self.block5 = DisBlock(args, self.ch*8, self.ch*16, activation=activation, downsample=True)
        self.block6 = DisBlock(args, self.ch*16, self.ch*16, activation=activation, downsample=False)
        self.l_y = nn.utils.spectral_norm(nn.Embedding(args.n_classes, self.ch * 16))
        self.l7 = nn.Linear(self.ch*16, 1, bias=False)
        if args.d_spectral_norm:
            self.l7 = nn.utils.spectral_norm(self.l7)

    def forward(self, x, y):
        h = x
        # print(h.shape)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        if y is not None:
            emb = self.embed(y)
            H, W = h.shape[2], h.shape[3]
            emb = emb.reshape(emb.shape[0], emb.shape[1], 1, 1)
            emb = emb.expand(emb.shape[0], emb.shape[1], H, W)
            h = torch.cat((h, emb), 1)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l7(h)
        return output
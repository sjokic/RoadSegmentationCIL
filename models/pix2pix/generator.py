'''
Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
and https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/pix2pix/pix2pix_module.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def concat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    return torch.cat([x1, x2], dim=1)


class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2) #LeakyReLU by default

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, features):
        """
        - Encoder: C64-C128-C256-C512-C512-C512
        - Convolutions use 4×4 spatial filters and a stride of 2
        - Decoder: CD512-CD1024-CD1024-C512-C256-C64
        """
        super().__init__()

        # best choice of number of downsample/upsample layers is 6
        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, features, batchnorm=False),
            DownSampleConv(features, features * 2),
            DownSampleConv(features * 2, features * 4),
            DownSampleConv(features * 4, features * 8),
            DownSampleConv(features * 8, features * 8),
            DownSampleConv(features * 8, features * 8),
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(features * 8, features * 8, dropout=True),
            UpSampleConv(features * 8 * 2, features * 8, dropout=True),
            UpSampleConv(features * 8 * 2, features * 4, dropout=True),
            UpSampleConv(features * 4 * 2, features * 2),
            UpSampleConv(features * 2 * 2, features),
        ]

        self.final_conv = nn.ConvTranspose2d(
            features, out_channels, kernel_size=4, stride=2, padding=1)

        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            x = concat(x, skip)

        x = self.decoders[-1](x)
        x = self.final_conv(x)

        return self.tanh(x)

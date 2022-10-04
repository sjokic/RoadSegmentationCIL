""" 
Parts of the LA-Net model https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102424 
Code adapted from https://github.com/ggsDing/LANet
"""

from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import models


class ConvUnit(pl.LightningModule):
    """ Convolution unit (conv2d -> bn -> relu) * 2 """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(pl.LightningModule):
    """ Max-pool followed by ConvUnit """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvUnit(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class PatchAttention(pl.LightningModule):
    """ 
    Patch attention module:

    reduction       encoded state has (in_channels // reduction) channels
    pool_window     window size for adaptive average pooling, defines patch size
    """

    def __init__(self, in_channels, reduction, pool_window, add_input=False):
        super().__init__()
        self.pool_window = pool_window
        self.add_input = add_input

        # Attention
        self.SA = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, h, w = x.size()
        pool_out_h = h // self.pool_window
        pool_out_w = w // self.pool_window

        A = F.adaptive_avg_pool2d(x, (pool_out_h, pool_out_w))
        A = self.SA(A)

        # Upsample pooled features back to input size
        A = F.upsample(A, (h, w), mode='bilinear')

        output = x * A
        if self.add_input:
            output += x

        return output


class AttentionEmbedding(pl.LightningModule):
    """
    Attention embedding module:

    reduction       encoded state has (in_channels // reduction) channels
    pool_window     window size for adaptive average pooling, defines patch size
    """

    def __init__(self, in_channels, out_channels, reduction, pool_window, add_input=False):
        super().__init__()
        self.add_input = add_input

        # Embedder
        self.SE = nn.Sequential(
            nn.AvgPool2d(kernel_size=pool_window+1,
                         stride=1, padding=pool_window//2),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, high_features, low_features):
        batch, channels, h, w = low_features.size()

        # Embed the high level features and upsample them to the size of the low level features
        A = self.SE(high_features)
        A = F.upsample(A, (h, w), mode='bilinear')

        # Mult to the low level features
        output = low_features * A
        if self.add_input:
            output += low_features

        return output


class ResNet18(pl.LightningModule):
    """
    ResNet 18 Encoder Backbone, from the LA-Net Source Code
    """

    def __init__(self, in_channels, out_channels, pretrained=True):
        super().__init__()

        resnet = models.resnet18(pretrained)

        # Modify the 1st conv so its consistent with the number of in_channels & transfer weights
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:in_channels, :, :].copy_(
            resnet.conv1.weight.data[:, 0:in_channels, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.head = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128, momentum=0.95),
                                  nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128, momentum=0.95),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x_size = x.size()

        x0 = self.layer0(x)     # size:1/2
        x = self.maxpool(x0)    # size:1/4
        x = self.layer1(x)      # size:1/4
        x = self.layer2(x)      # size:1/8
        x = self.layer3(x)      # size:1/16
        # x = self.layer4(x)
        x = self.head(x)
        x = self.classifier(x)

        return F.upsample(x, x_size[2:], mode='bilinear')

""" 
Model which adapts the LA-Net https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102424 
Code adapted from https://github.com/ggsDing/LANet
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from .lanet_parts import *


class LANet(pl.LightningModule):
    """
    LANet model from "LANet: Local Attention Embedding to Improve the Semantic Segmentation 
    of Remote Sensing Images". Modified to take advantage of patch sizes for prediction.
    """

    def __init__(self, n_channels, n_filters, n_classes, loss_fn, metric,
                 bilinear=True, pam=True, aem=True, lr=1e-4):
        """
        Init model. Can decide whether or not to enable the PAM/AEM.
            - Removing the PAM just passes the raw low/high level features
            - Removing the AEM upsamples and concatenates the high level features to the low
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.loss_fn = loss_fn
        self.metric = metric
        self.bilinear = bilinear
        self.lr = lr
        self.pam = pam
        self.aem = aem

        self.inc = ConvUnit(self.n_channels, n_filters)
        self.down1 = Down(n_filters, n_filters * 2)             # size 1/2
        self.down2 = Down(n_filters * 2, n_filters * 4)         # size 1/4
        self.down3 = Down(n_filters * 4, n_filters * 8)         # size 1/8
        self.down4 = Down(n_filters * 8, n_filters * 16)        # size 1/16

        # Head, retain size and n_filters * 16 -> n_filters
        self.head = nn.Sequential(
            nn.Conv2d(n_filters * 16, n_filters, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters, momentum=0.95),
            nn.ReLU()
        )

        # PAM at low level feature space
        # Apply Conv unit regardless of PAM
        # Apply this after the first down layer, hence n_filters * 2 -> n_filters // 2
        self.low_conv = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters // 2, kernel_size=1),
            nn.BatchNorm2d(n_filters // 2, momentum=0.95),
            nn.ReLU(inplace=False)
        )
        self.patch_attn_low = PatchAttention(n_filters // 2, reduction=4,
                                             pool_window=16, add_input=True)

        # PAM at high level feature space, less pooling and more reduction
        # Apply this after the head, hence n_filters * 2
        self.patch_attn_high = PatchAttention(n_filters, reduction=16,
                                              pool_window=4, add_input=True)

        # AEM between PAM_HIGH -> PAM_LOW
        self.attn_embedding = AttentionEmbedding(
            n_filters, n_filters // 2, reduction=8, pool_window=3)

        # Classifier for low path
        if self.aem:
            self.classifier_low = nn.Conv2d(n_filters // 2, n_classes,
                                            kernel_size=1)
        else:
            # More input features without AEM (since features are concatenated)
            self.classifier_low = nn.Conv2d(n_filters + n_filters // 2, n_classes,
                                            kernel_size=1)

        # Classifier for high path
        self.classifier_high = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()

        x = self.inc(x)
        x_low = self.down1(x)
        x = self.down2(x_low)
        x = self.down3(x)
        x = self.down4(x)
        x_high = self.head(x)

        # Prepare low level features for PAM
        x_low = self.low_conv(x_low)

        # PAMs
        if self.pam:
            x_low = self.patch_attn_low(x_low)
            x_high = self.patch_attn_high(x_high)

        # AEM
        if self.aem:
            x_low = self.attn_embedding(x_high.detach(), x_low)
        else:
            x_high = F.upsample(x_high, (x_low.size()[2], x_low.size()[3]), mode='bilinear')
            x_low = torch.cat([x_high, x_low], dim=1)

        # Classify
        x_low = self.classifier_low(x_low)
        x_low = F.upsample(x_low, x_size[2:], mode='bilinear')

        x_high = self.classifier_high(x_high)
        x_high = F.upsample(x_high, x_size[2:], mode='bilinear')

        return x_low + x_high

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Optimizer is configured to use Adam with custom LR.
        Default uses 1e-8 weight decay TODO: Configure to allow different values.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr, weight_decay=1e-8)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """ Validates using given metric function """
        x, y = batch
        preds = self(x)

        score = self.metric(preds, y)
        self.log('val_{}'.format(self.metric.name), score)


class LANetResnet(pl.LightningModule):
    """
    LANet model from "LANet: Local Attention Embedding to Improve the Semantic Segmentation 
    of Remote Sensing Images". From the original LA-Net source code, with a pooling layer removed.
    """

    def __init__(self, n_channels, n_filters, n_classes, loss_fn, metric, bilinear=True, lr=1e-4):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.loss_fn = loss_fn
        self.metric = metric
        self.bilinear = bilinear
        self.lr = lr

        self.resnet = ResNet18(n_channels, n_classes)
        self.PA0 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                 nn.BatchNorm2d(64, momentum=0.95), nn.ReLU(inplace=False),
                                 PatchAttention(64, reduction=8, pool_window=20, add_input=True))

        self.PA2 = PatchAttention(128, reduction=16, pool_window=4, add_input=True)
        self.AE = AttentionEmbedding(128, 64, reduction=16, pool_window=6)

        self.classifier0 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.classifier1 = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()

        x = self.resnet.layer0(x)   # size:1/2
        x = self.resnet.maxpool(x)  # size:1/4
        x0 = self.resnet.layer1(x)  # size:1/4  C64
        x = self.resnet.layer2(x0)  # size:1/8  C128
        x = self.resnet.layer3(x)   # size:1/16 C256
        x2 = self.resnet.head(x)    # size:1/16 C128

        x2 = self.PA2(x2)
        x0 = self.PA0(x0)
        x0 = self.AE(x2.detach(), x0)

        low = self.classifier0(x0)
        low = F.upsample(low, x_size[2:], mode='bilinear')

        high = self.classifier1(x2)
        high = F.upsample(high, x_size[2:], mode='bilinear')

        return high+low  # , high , low

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Optimizer is configured to use Adam with custom LR.
        Default uses 1e-8 weight decay TODO: Configure to allow different values.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr, weight_decay=1e-8)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """ Validates using given metric function """
        x, y = batch
        preds = self(x)

        score = self.metric(preds, y)
        self.log('val_{}'.format(self.metric.name), score)

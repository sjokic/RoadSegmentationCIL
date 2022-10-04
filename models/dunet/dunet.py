"""
U-Net model with dilation block.

Code adapted from https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from models.unet.unet_parts import *
from models.dinknet.dinknet import Dblock


class DUNet(pl.LightningModule):
    def __init__(self, n_channels, n_filters, n_classes, loss_fn, metric, bilinear=True, lr=1e-4):
        super(DUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.loss_fn = loss_fn
        self.metric = metric
        self.bilinear = bilinear
        self.lr = lr

        # Initial convolution
        self.inc = DoubleConv(self.n_channels, n_filters)

        # Down passes
        self.down1 = Down(n_filters, n_filters * 2)
        self.down2 = Down(n_filters * 2, n_filters * 4)
        self.down3 = Down(n_filters * 4, n_filters * 8)
        self.down4 = Down(n_filters * 8, n_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_filters * 8, (n_filters * 16) // factor)

        # Dilation block
        self.dblock = Dblock((n_filters * 16) // factor)

        self.up1 = Up(n_filters * 16, (n_filters * 8) // factor, bilinear)
        self.up2 = Up(n_filters * 8, (n_filters * 4) // factor, bilinear)
        self.up3 = Up(n_filters * 4, (n_filters * 2) // factor, bilinear)
        self.up4 = Up(n_filters * 2, n_filters, bilinear)
        self.outc = OutConv(n_filters, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.dblock(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

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

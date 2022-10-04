""" UNet Model, adapted from https://github.com/milesial/Pytorch-UNet for pytorch lightning """

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from .unet_parts import *


class UNet(pl.LightningModule):
    """Full assembly of all parts of the UNet"""

    def __init__(self, n_channels, n_filters, n_classes, loss_fn, metric, bilinear=True, lr=1e-4):
        """
        Initialize a UNet to predict images with n_channels to n_classes.
        Loss function should take a raw module output and apply sigmoid before computing loss.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.loss_fn = loss_fn
        self.metric = metric
        self.bilinear = bilinear
        self.lr = lr

        self.inc = DoubleConv(self.n_channels, n_filters)
        self.down1 = Down(n_filters, n_filters * 2)
        self.down2 = Down(n_filters * 2, n_filters * 4)
        self.down3 = Down(n_filters * 4, n_filters * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(n_filters * 8, (n_filters * 16) // factor)
        self.up1 = Up(n_filters * 16, (n_filters * 8) // factor, bilinear)
        self.up2 = Up(n_filters * 8, (n_filters * 4) // factor, bilinear)
        self.up3 = Up(n_filters * 4, (n_filters * 2) // factor, bilinear)
        self.up4 = Up(n_filters * 2, n_filters, bilinear)
        self.outc = OutConv(n_filters, self.n_classes)

        # Save predictions for later
        self.predictions = []

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
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

    def test_step(self, batch, batch_idx):
        """ Predict and save predicted masks to self.pred_out_dir. """
        x, img_name = batch
        preds = self(x)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).int()

        transform = transforms.Compose([
            transforms.ToPILImage(),
        ])

        for mask in preds:
            img = transform(mask)
            self.predictions.append(img)

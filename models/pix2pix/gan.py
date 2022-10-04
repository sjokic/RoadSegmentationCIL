'''
Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py
and https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/pix2pix/pix2pix_module.py
'''

import torch
import torch.nn as nn
import pytorch_lightning as pl
from .generator import UNet
from .discriminator import PatchGAN
from torchvision import transforms
from metrics import IoU


def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Pix2Pix(pl.LightningModule):

    # best choice for lr so far: 6e-4, 2e-4
    def __init__(self, in_channels=3, out_channels=1, learning_rate=[6e-4, 2e-4], lambda_recon=200, metric=None):

        super().__init__()
        self.save_hyperparameters()

        # self.metric = metric
        self.metric = IoU(0.5, activation=nn.Identity())

        self.gen = UNet(in_channels, out_channels, features=64)
        self.patch_gan = PatchGAN(in_channels + out_channels)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

        self.sigmoid = nn.Sigmoid()

        self.num_trainingstep = 0

    def reset_discriminator(self):
        self.patch_gan = self.patch_gan.apply(_weights_init)

    def forward(self, x):
        # last activation layer is tanh so map back to [0, 1]
        return 0.5*(self.gen(x)+1)

    def _gen_step(self, imgs, lbls):
        # calculate the adversarial loss
        fake_lbls = 0.5*(self.gen(imgs)+1)
        disc_logits = self.patch_gan(fake_lbls, imgs)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_criterion(fake_lbls, lbls)
        lambda_recon = self.hparams.lambda_recon

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, imgs, lbls):
        fake_lbls = 0.5*(self.gen(imgs)+1)
        fake_logits = self.patch_gan(fake_lbls, imgs)

        real_logits = self.patch_gan(lbls, imgs)

        # one-sided label smoothing to make sure discriminator doesn't become overconfident (see https://arxiv.org/pdf/1606.03498.pdf)
        smoothed_labels = torch.ones_like(fake_logits)
        smoothed_labels[smoothed_labels == 1.0] = 0.9

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, smoothed_labels)
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr[0], betas=(0.5, 0.999))
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=lr[1], betas=(0.5, 0.999))  # lr=5e-5 previously

        return [disc_opt, gen_opt]

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, lbls = batch

        loss = None

        if optimizer_idx == 0 and self.num_trainingstep == 2:
            loss = self._disc_step(imgs, lbls)
            self.log('Discriminator Loss', loss)
            print('Discriminator loss: ', loss.item())
        elif optimizer_idx == 1:
            loss = self._gen_step(imgs, lbls)
            self.log('Generator Loss', loss)
            print('Generator loss: ', loss.item())

        # Training of the discriminator is skipped every two training steps
        self.num_trainingstep += 1
        if(self.num_trainingstep > 2):
            self.num_trainingstep = 0

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = 0.5*(self.gen(x)+1)

        score = self.metric(preds, y)
        self.log('val_{}'.format(self.metric.name), score)

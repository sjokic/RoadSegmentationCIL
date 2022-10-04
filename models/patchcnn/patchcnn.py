"""
Patch CNN, adapted from
https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_3.ipynb
for pytorch lightning
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class PatchCNN(pl.LightningModule):

    def __init__(self, n_channels=3, n_filters=0, n_classes=1, loss_fn=None, metric=None, lr=1e-3):
        super().__init__()

        self.loss_fn = loss_fn
        self.metric = metric
        self.lr = lr

        self.net = nn.Sequential(nn.Conv2d(in_channels=n_channels, out_channels=16, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(16),
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(64),
                                 nn.Dropout(0.5),
                                 nn.Flatten(),
                                 nn.Linear(256, 10),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(10, n_classes))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.net(x)

        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Optimizer is configured to use Adam with custom LR.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        """ Validates using given metric function """
        x, y = batch
        preds = torch.sigmoid(self.net(x))

        #score = self.metric(preds, y)
        score = (preds.round() == y.round()).float().mean()
        self.log('val_{}'.format('accuracy'), score)

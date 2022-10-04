""" Custom metrics. All take logits and targets. """

from typing import IO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def threshold(preds, threshold):
    """
    Generates binary predictions based on a threshold.
    """
    return (preds > threshold).int()


class Accuracy(nn.Module):
    """
    Pixel-wise accuracy.
    """
    name = 'accuracy'

    def __init__(self, threshold=0.5, activation=torch.sigmoid):
        super().__init__()
        self.threshold = threshold
        self.activation = activation

    def forward(self, logits, targets):
        preds = self.activation(logits)
        preds = threshold(preds, self.threshold)

        correct = (preds == targets).sum()
        accuracy = correct / targets.view(-1).shape[0]

        return accuracy


class IoU(nn.Module):
    """ Jaccard, i.e., IoU """
    name = 'iou'
    SMOOTH = 1e-7

    def __init__(self, threshold=0.5, activation=torch.sigmoid):
        super().__init__()
        self.threshold = threshold
        self.activation = activation

    def forward(self, logits, targets):
        preds = self.activation(logits)
        preds = threshold(preds, self.threshold)

        intersection = torch.sum(preds * targets)
        union = torch.sum(preds) + torch.sum(targets) - intersection

        return (intersection + self.SMOOTH) / (union + self.SMOOTH)


metrics_index = {
    'accuracy': Accuracy(),
    'iou': IoU(),
    'jaccard': IoU()
}

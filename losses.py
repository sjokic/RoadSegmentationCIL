""" Custom losses. """

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.BCEWithLogitsLoss):
    """ Just a reskin of the standard BCEWithLogitsLoss """
    name = 'bce'


class DiceLoss(nn.Module):
    """ Pure dice loss with a smoothing term to prevent divide by 0 """
    name = 'dice'
    SMOOTH = 1e-7

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)

        n_classes = targets.size(0)
        preds = preds.view(n_classes, -1)
        targets = targets.view(n_classes, -1)

        intersection = torch.sum(preds * targets, 1)
        cardinality = torch.sum(preds + targets, 1)

        dice = (2. * intersection + self.SMOOTH) / (cardinality + self.SMOOTH)
        return (1. - dice).mean()


class JaccardLoss(nn.Module):
    """ Jaccard (IoU) Loss """
    name = 'jaccard'
    SMOOTH = 1e-7

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)

        n_classes = targets.size(0)
        preds = preds.view(n_classes, -1)
        targets = targets.view(n_classes, -1)

        intersection = torch.sum(preds * targets, 1)
        cardinality = torch.sum(preds + targets, 1)
        union = cardinality - intersection

        jaccard = (intersection + self.SMOOTH) / (union + self.SMOOTH)
        return (1. - jaccard).mean()


class FocalLoss(nn.Module):
    """ Focal Loss """
    name = 'focal'

    def __init__(self, alpha=0.25, gamma=2):
        """
        alpha:  the prior probability of positive pixel in target
        gamma:  dampening weight / focal strength
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits)

        n_classes = targets.size(0)
        preds = preds.view(n_classes, -1)
        targets = targets.view(n_classes, -1)

        targets = targets.type(preds.type())
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        bce_exp = torch.exp(-bce)

        focal_term = (1.0 - bce_exp).pow(self.gamma)
        focal_loss = focal_term * bce

        focal_loss *= self.alpha * targets + (1 - self.alpha) * (1 - targets)

        return focal_loss.mean()


class CompositeLoss(nn.Module):
    """
    Takes a list of (Loss(), weight) tuples and calculates the weighted sum of those losses.
    """

    def __init__(self, name, components):
        super().__init__()

        self.name = name
        self.components = components

    def forward(self, logits, targets):
        losses = [loss.forward(logits, targets) * weight for (loss, weight) in self.components]
        return sum(losses)


losses_index = {
    'bce': BCELoss(),
    'dice': DiceLoss(),
    'jaccard': JaccardLoss(),
    'focal': FocalLoss(),
    'bce-jacc': CompositeLoss('bce-jacc', [
        (BCELoss(), 1.0),
        (JaccardLoss(), 1.0)
    ]),
    'focal-jacc': CompositeLoss('focal-jacc', [
        (FocalLoss(), 1.0),
        (JaccardLoss(), 1.0)
    ]),
    'bce-dice': CompositeLoss('bce-dice', [
        (BCELoss(), 1.0),
        (DiceLoss(), 1.0)
    ])
}

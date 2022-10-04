""" Programmatically get model instances from their names """

from .unet.unet_model import UNet
from .lanet.lanet_model import LANet, LANetResnet
from .pix2pix.gan import Pix2Pix
from .dinknet.dinknet import DinkNet
from .dunet.dunet import DUNet
from .patchcnn.patchcnn import PatchCNN

model_index = {
    'patchcnn' : PatchCNN,
    'unet': UNet,
    'lanet': LANet,
    'lanet-res': LANetResnet,
    'pix2pix': Pix2Pix,
    'dinknet': DinkNet,
    'dunet': DUNet
}


def get_model(model, n_channels, n_filters, n_classes, loss_fn, metric, lr):
    """
    Return an instance of a model given a name string and args.
    """
    if (model == 'pix2pix'):
        return model_index[model](in_channels=n_channels, out_channels=n_classes, metric=metric)

    return model_index[model](n_channels, n_filters, n_classes, loss_fn, metric, lr)


def get_model_from_ckpt(model, path_to_ckpt, n_channels, n_filters, n_classes,
                        loss_fn, metric, lr=1e-4):
    """
    Load pre-trained weights from a checkpoint file for the model specified in 'model'.
    """
    if (model == 'pix2pix'):
        return model_index[model].load_from_checkpoint(path_to_ckpt, in_channels=n_channels, out_channels=n_classes, metric=metric)

    return model_index[model].load_from_checkpoint(path_to_ckpt, n_channels=n_channels,
                                                   n_filters=n_filters, n_classes=n_classes,
                                                   loss_fn=loss_fn, metric=metric, lr=lr)

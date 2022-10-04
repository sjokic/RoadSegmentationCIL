"""
Patch CNN data module, adapted from
https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_3.ipynb
for pytorch lightning
"""

import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

PATCH_SIZE = 16  # pixels per side of square patches
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road

# I pre-computed the means and stds of the given Kaggle SatImage training data
# Values are relative to torch.Tensor image range, i.e., pixel values \in [0, 1]
# TODO: Make this dynamic? Is that even necessary?
MEANS = (0.3329814, 0.33009395, 0.29579765)
STDS = (0.19390681, 0.18714972, 0.18701904)

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)

def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.

def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % PATCH_SIZE) + (w % PATCH_SIZE) == 0  # make sure images can be patched exactly

    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE
    patches = images.reshape((n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape((n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels

class PatchImageDataset(Dataset):
    """
    Torch Dataset for Kaggle road seg training data.
    Generates patches for each image.
    """

    def __init__(self, img_dir, lbl_dir, normalize=True, gpus=0, use_patches=True, resize_to=(400, 400)):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.device = 'cuda' if (torch.cuda.is_available() and gpus > 0) else 'cpu'

        self.norm = transforms.Normalize(MEANS, STDS) if normalize else None

        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(self.img_dir)
        self.y = load_all_from_path(self.lbl_dir)

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def __len__(self):
        return self.n_samples

    def _preprocess(self, x, y):
        # EDIT: Do not normalize. The computed means and stds are not per patch.
        # Just normalize if flag is set
        # if self.norm is not None:
        #     x = self.norm(x)

        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))

class PatchImageTestDataset(Dataset):
    """
    Test Dataset for Kaggle road seg training data.
    Generates patches for each image.
    """

    def __init__(self, img_dir, normalize=True, gpus=0, use_patches=True, resize_to=(400, 400)):
        self.img_dir = img_dir

        self.device = 'cuda' if (torch.cuda.is_available() and gpus > 0) else 'cpu'

        self.norm = transforms.Normalize(MEANS, STDS) if normalize else None

        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.n_samples = None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(self.img_dir)
        if self.use_patches:  # split each image into patches
            self.x = image_to_patches(self.x)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)

        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def __len__(self):
        return self.n_samples

    def _preprocess(self, x):
        # Just normalize if flag is set

        # if self.norm is not None:
        #     x = self.norm(x)

        return x

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device))


class SatImagePatchesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Kaggle road seg data.
    """

    def __init__(self, train_img_dir, train_lbl_dir, test_img_dir, batch_size, val_size=0.1,
                 n_workers=0, normalize=True, gpus=0):
        """
        Initializes an instance of this DataModule. Params:
            - train_img_dir, train_lbl_dir, test_img_dir: paths to respective data directories
            - batch_size: batch size to be used for all stages
            - val_size: percentage of samples for validation set
            - n_workers: number of CPU workers to read data
            - data_aug: whether or not to apply data augmentation
            - normalize: whether or not to apply normalization
        """
        super().__init__()

        # Store directory paths
        self.dirs = {
            'train_img': train_img_dir,
            'train_lbl': train_lbl_dir,
            'test_img': test_img_dir
        }

        # Set params trivially
        self.batch_size = batch_size
        self.val_size = val_size
        self.n_workers = n_workers
        self.normalize = normalize
        self.gpus = gpus

        # Image dimensions statically set
        self.dims = (3, 400, 400)

    def setup(self, stage=None):
        """
        Load data, create val split, and apply transforms.
        The stage param is automatically passed by pytorch lightning for 'fit'/'test'.
        """
        # Load all the data
        if stage == 'fit' or stage is None:

            # Create a Dataset instance
            data_full = PatchImageDataset(img_dir=self.dirs['train_img'], lbl_dir=self.dirs['train_lbl'], normalize=self.normalize, gpus=self.gpus)

            # Compute split
            val_length = int(len(data_full) * self.val_size)
            train_length = int(len(data_full) - val_length)
            self.train, self.val = random_split(
                data_full, [train_length, val_length], generator=torch.Generator().manual_seed(42))

        if stage == 'test' or stage is None:
            self.test = PatchImageTestDataset(img_dir=self.dirs['test_img'], normalize=self.normalize, gpus=self.gpus)

    # Return the dataloader for each split.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.n_workers)

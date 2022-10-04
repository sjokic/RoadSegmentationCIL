""" DataModule for DeepGlobe data """

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

from .cilab import SatImageDataset, MapDataset, TRANSFORM


class DeepGlobeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for cropped DeepGlobe data.
    """

    def __init__(self, train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir,
                 batch_size, val_size=0.2, n_workers=0):
        """
        Needs training directory (deepglobe) as well as the actual competition training directory.
        """
        super().__init__()

        # Store directory paths
        self.dirs = {
            'train_img': train_img_dir,
            'train_lbl': train_lbl_dir,
            'val_img': val_img_dir,
            'val_lbl': val_lbl_dir
        }

        # Set params trivially
        self.batch_size = batch_size
        self.val_size = val_size
        self.n_workers = n_workers

    def setup(self, stage=None):
        """
        Load data and create val split.
        The stage param is automatically passed by pytorch lightning for 'fit'/'test'.
        """
        # Load all the data
        if stage == 'fit' or stage is None:
            # Create a SatImageDataset instance for regular training
            train_set = SatImageDataset(self.dirs['train_img'], self.dirs['train_lbl'])

            # Use same validation set as regular training
            cilab_data = SatImageDataset(self.dirs['val_img'], self.dirs['val_lbl'])

            # Find the same split as the regular training
            val_length = int(len(cilab_data) * 0.2)
            discarded_length = int(len(cilab_data) - val_length)

            _, val_subset = random_split(cilab_data, [discarded_length, val_length],
                                         generator=torch.Generator().manual_seed(42))

            # Assign transforms and set train/val datasets using MapDataset
            self.train = MapDataset(train_set, TRANSFORM)
            self.val = MapDataset(val_subset, TRANSFORM)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.n_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.n_workers)

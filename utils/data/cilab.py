""" Code necessary to load the dataset(s). """

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# I pre-computed the means and stds of the given Kaggle SatImage training data
# Values are relative to torch.Tensor image range, i.e., pixel values \in [0, 1]
MEANS = [0.3329814, 0.33009395, 0.29579765]
STDS = [0.19390681, 0.18714972, 0.18701904]

# What value in [0, 255] should we threshold the masks?
MASK_THRESHOLD = 0.5 * 255

# Define your transform with augmentation here.
AUGMENTED_TRANSFORM = A.Compose([
    A.ShiftScaleRotate(
        p=0.5,
        scale_limit=0.1,
        shift_limit=0.0625,
        rotate_limit=45,
        border_mode=cv2.BORDER_REFLECT_101,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=MEANS, std=STDS),
    ToTensorV2()
])

AUGMENTED_TRANSFORM_PIX2PIX = A.Compose([
    A.RandomResizedCrop(400, 400, scale=(0.35, 1.0), p=0.5),
    A.ShiftScaleRotate(
        p=0.5,
        scale_limit=0,
        shift_limit=0.0625,
        rotate_limit=90,
        border_mode=cv2.BORDER_REFLECT_101,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=MEANS, std=STDS),
    ToTensorV2()
])

# Default transform, without augmentation flag.
TRANSFORM = A.Compose([
    A.Normalize(mean=MEANS, std=STDS),
    ToTensorV2()
])


class MapDataset(Dataset):
    """
    Wrapper class so that we can apply different albumentations transforms to the training
    and validation dataset.

    The provided inner dataset should return: image, mask
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return image, mask.unsqueeze(0)


class SatImageDataset(Dataset):
    """
    Torch Dataset for Kaggle road seg training data.
    """

    def __init__(self, image_dir, mask_dir):
        """
        Initialize this Dataset instance.
            - image_dir, mask_dir: paths to directories containing images and masks
        """
        self.image_paths = []
        self.mask_paths = []

        # Get sorted file names in image and mask dir (to align just in case)
        image_paths = sorted(os.listdir(image_dir))
        mask_paths = sorted(os.listdir(mask_dir))

        for image_path, mask_path in zip(image_paths, mask_paths):
            # Skip any .gitignores or .DS_Stores (that rhymed btw)
            if image_path[0] == '.' or mask_path[0] == '.':
                continue

            # Assert paths are the same
            assert(image_path == mask_path)

            # Append paths
            self.image_paths.append(os.path.join(image_dir, image_path))
            self.mask_paths.append(os.path.join(mask_dir, mask_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads the .PNG image, converts to RGB and polarizes the mask.
        NOTE: Does not convert to tensor or normalize -> deferred to MapDataset!
        """
        # Process image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process label
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > MASK_THRESHOLD).astype(np.float32)       # Polarize the mask

        return image, mask


class SatImageTestDataset(Dataset):
    """
    Torch Dataset of Kaggle road seg test images.
    """

    def __init__(self, image_dir, transform=None):
        # Generate set of image paths in the directory
        self.image_paths = []
        self.image_dir = image_dir

        for image_path in os.listdir(image_dir):
            if image_path[0] == '.':
                continue

            self.image_paths.append(image_path)

        # Normalize and to tensor transform
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, self.image_paths[idx]


class SatImageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Kaggle road seg data.
    """

    def __init__(self, train_img_dir, train_mask_dir, test_img_dir, batch_size, val_size=0.2,
                 n_workers=0, data_aug=False, model=None):
        """
        Initializes an instance of this DataModule. Params:
            - train_img_dir, train_mask_dir, test_img_dir: paths to respective data directories
            - batch_size: batch size to be used for all stages
            - val_size: percentage of samples for validation set
            - n_workers: number of CPU workers to read data
            - data_aug: whether or not to apply data augmentation
        """
        super().__init__()

        # Store directory paths
        self.dirs = {
            'train_img': train_img_dir,
            'train_mask': train_mask_dir,
            'test_img': test_img_dir
        }

        # Set params trivially
        self.batch_size = batch_size
        self.val_size = val_size
        self.n_workers = n_workers
        self.data_aug = data_aug
        self.model = model

    def setup(self, stage=None):
        """
        Load data, create val split, and apply transforms.
        The stage param is automatically passed by pytorch lightning for 'fit'/'test'.
        """
        # Load all the data
        if stage == 'fit' or stage is None:
            # Create a full Dataset instance
            data_full = SatImageDataset(self.dirs['train_img'], self.dirs['train_mask'])

            # Compute the train-val split
            val_length = int(len(data_full) * self.val_size)
            train_length = int(len(data_full) - val_length)

            # Get two subset datasets according to the random split
            train_subset, val_subset = random_split(data_full, [train_length, val_length],
                                                    generator=torch.Generator().manual_seed(42))

            # Set the transforms (with augmentation if desired)
            aug_transforms = AUGMENTED_TRANSFORM_PIX2PIX if self.model=='pix2pix' else AUGMENTED_TRANSFORM

            self.train = MapDataset(
                train_subset, aug_transforms if self.data_aug else TRANSFORM)
            self.val = MapDataset(val_subset, TRANSFORM)

        if stage == 'test' or stage is None:
            self.test = SatImageTestDataset(self.dirs['test_img'], transform=TRANSFORM)

    # Return the dataloader for each split.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.n_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.n_workers)

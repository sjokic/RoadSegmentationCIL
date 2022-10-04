""" Script to generate probabilistic masks on the test set using a checkpoint file. """

import sys
from albumentations import augmentations
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

import os
import re
from itertools import combinations, chain
from argparse import ArgumentParser

from metrics import IoU
from models.models import get_model_from_ckpt, model_index
from utils.data.cilab import SatImageDataModule, MEANS, STDS

from models.patchcnn.predict import patch_predict

# Local paths to directories with training data
TRAIN_IMG_DIR = 'training/training/images/'
TRAIN_LBL_DIR = 'training/training/groundtruth/'
TEST_IMG_DIR = 'test_images/test_images/'

# Define a list of augmentations (function, forward_kwargs, reverse_kwargs)
AUGMENTATIONS = [
    (F.hflip, {}, {}),
    (F.vflip, {}, {}),
    (F.rotate, {'angle': 90}, {'angle': -90}),
    (F.rotate, {'angle': 180}, {'angle': -180}),
    (F.rotate, {'angle': 270}, {'angle': -270})
]

# To Pillow Image transform
TO_PILLOW = transforms.ToPILImage()


def predict(image, model, args):
    """
    Given an Tensor image and a pretrained model: make a prediction and return it
    as a Pillow Image.
    """
    preds = model(image.unsqueeze(0))
    if(args.model == 'pix2pix'):
        preds = preds[0]
    else:
        preds = torch.sigmoid(preds)[0]

    return TO_PILLOW(preds)


def tta_predict(image, model, args):
    """
    Given an NPARRAY image, a pretrained model, and a list of augmentations,
    apply all combinations of those augments and average the predictions,
    then return the averaged prediction as a Pillow Image.
    """
    # Generate aug_sets as an iterable of (FORWARD, BACKWARD) transform tuples
    if args.aug_mode == 'powerset':
        # Power-set -> generate powerset of augmentations
        aug_sets = chain.from_iterable(
            combinations(AUGMENTATIONS, r) for r in range(len(AUGMENTATIONS)+1))
    elif args.aug_mode == 'independent':
        # Independent -> all augmentations by themselves, plus no augmentation
        aug_sets = [[]] + [[aug] for aug in AUGMENTATIONS]

    # Want to keep an array of preds
    preds = []

    # Go through each set of augs
    for augs in aug_sets:
        # Perform forward transformation
        transformed = image.unsqueeze(0)

        for func, forward_args, _ in augs:
            transformed = func(transformed, **forward_args)

        # Make prediction
        if(args.model == 'pix2pix'):
            pred = model(transformed)[0]
        else:
            pred = torch.sigmoid(model(transformed))[0]


        # Perform backward transformation
        for func, _, backward_args in reversed(augs):
            pred = func(pred, **backward_args)

        # Add to preds array
        preds.append(pred)

    # Return the averaged prediction as a Pillow image
    # (only way I could get this working was np -> torch -> Pillow)
    pred = torch.cat(preds)
    pred = torch.mean(pred, axis=0)
    return TO_PILLOW(pred)


def main(args):
    # Make directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Initialize an instance of the DataModule for Kaggle road seg data
    dm = SatImageDataModule(TRAIN_IMG_DIR, TRAIN_LBL_DIR,
                            TEST_IMG_DIR, 1, n_workers=args.n_workers)
    dm.setup('test')

    # Load and freeze model with random loss function (doesn't matter)
    print('Loading checkpoint file...')
    pretrained_model = get_model_from_ckpt(args.model, args.ckpt, n_channels=3,
                                           n_filters=args.n_filters, n_classes=1,
                                           loss_fn=nn.BCEWithLogitsLoss(), metric=IoU())
    pretrained_model.freeze()

    # Use GPU if the flag is set
    if args.gpu:
        device = torch.device("cuda")
        pretrained_model.to(device)

    # If model is PatchCNN, directly classify patches and terminate
    if args.model == 'patchcnn':
        patch_predict(pretrained_model, TEST_IMG_DIR, args.out_dir, args.gpu)
        sys.exit()

    # Get dataset so that we can make predictions
    test_dataset = dm.test

    # Save prediction image filepaths for the submission method
    pred_paths = []

    # Choose prediction function given flag
    predict_fn = tta_predict if args.augmentation else predict

    # Iterate through all the prediction images
    with tqdm(total=len(test_dataset), desc='Make predictions', unit='image') as pbar:
        # Since the test_dataloader is batched (even though BS always 1)
        for idx in range(len(test_dataset)):
            image, image_name = test_dataset[idx]

            if args.gpu:
                image = image.to(device='cuda')

            # Make prediction
            pred = predict_fn(image, pretrained_model, args)

            # Save the file
            name = image_name.split('/')[-1]
            pred_path = os.path.join(args.out_dir, name)

            pred.save(pred_path)
            pred_paths.append(pred_path)

            pbar.update()

    # Find the val_score and save it as a weight.txt file which can be used for a weighted ensemble
    matches = re.search(r"val_\w+=(\d.\d+)", args.ckpt)

    # Maybe could be more robust
    if matches is not None:
        val_score = matches.group(1)

        with open(os.path.join(args.out_dir, 'weight.txt'), 'w') as f:
            f.write(val_score)


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Output prediction images using a model checkpoint.')

    parser.add_argument('model', metavar='MODEL_NAME', choices=list(model_index.keys()),
                        help='Name of the ML model to use.')

    parser.add_argument('--ckpt', required=True,
                        help='Pytorch checkpoint file that contains model weights')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for prediction images')

    parser.add_argument('--n_workers', default=0, type=int,
                        help='Number of workers for DataLoaders')

    parser.add_argument('--augmentation', action='store_true',
                        help='Perform test time augmentation.')
    parser.add_argument('--aug_mode', choices=['independent', 'powerset'],
                        default='powerset', help='How to combine the augmentations during TTA.')

    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU.')

    # Network architecture hyperparameters
    parser.add_argument('--n_filters', default=32, type=int,
                        help='Number of filters of the first convolutional layer. The number of filters for the remaining layers are a multiple of n_filters.')
    args = parser.parse_args()

    main(args)

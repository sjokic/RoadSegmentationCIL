""" Take images from the DeepGlobe dataset and generate random 608x608 px crops. """

import os
from argparse import ArgumentParser

import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm


CROP_SIZE = 400


def get_random_crop(sat, mask):
    """
    Given two nparrays with shape (H, W, C), return a (CROP_SIZE, CROP_SIZE, C) random subarray of
    both (subarray indices are same for both nparrays, i.e., crop is consistent)
    """
    assert(sat.shape[0] == mask.shape[0] and sat.shape[1] == mask.shape[1])
    start_x = np.random.randint(sat.shape[0] - CROP_SIZE)
    start_y = np.random.randint(sat.shape[1] - CROP_SIZE)

    crop_sat = sat[start_x:start_x + CROP_SIZE, start_y:start_y + CROP_SIZE]
    crop_mask = mask[start_x:start_x + CROP_SIZE, start_y:start_y + CROP_SIZE]

    return crop_sat, crop_mask


def main(args):
    # Set numpy random seed for transforms
    np.random.seed(42)

    # Make output directories
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    sat_dir = os.path.join(args.out_dir, 'images')
    mask_dir = os.path.join(args.out_dir, 'groundtruth')

    if not os.path.exists(sat_dir):
        os.mkdir(sat_dir)

    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    # Get all image paths and match image to mask
    sats = []
    masks = []

    for f in os.listdir(args.data_dir):
        if f.endswith('_sat.jpg'):
            sats.append(f)
        elif f.endswith('_mask.png'):
            masks.append(f)

    # Sort and zip
    pairs = zip(sorted(sats), sorted(masks))

    # Go through each pair
    for sat_f, mask_f in tqdm(list(pairs)):
        assert(sat_f[:-len('_sat.jpg')] == mask_f[:-len('_mask.png')])
        sample_id = int(sat_f[:-len('_sat.jpg')])

        # Read images, converting the mask to a black and white
        sat = np.array(Image.open(os.path.join(args.data_dir, sat_f)))
        mask = np.array(Image.open(os.path.join(
            args.data_dir, mask_f)).convert('L'))
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

        for i in range(args.n):
            crop_sat, crop_mask = get_random_crop(sat, mask)

            # Save the crop
            name = 'dg_{}_{}.png'.format(sample_id, i)

            # Save original as RGB image
            Image.fromarray(crop_sat, 'RGB').save(os.path.join(sat_dir, name))

            # Save mask as black and white
            Image.fromarray(crop_mask[:, :, 0], 'L').save(
                os.path.join(mask_dir, name))

        if args.delete:
            # Delete original images for space considerations
            os.remove(os.path.join(args.data_dir, sat_f))
            os.remove(os.path.join(args.data_dir, mask_f))


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Generate random 608x608 crops of the DeepGlobe training data.')

    parser.add_argument('data_dir', metavar='DATA_DIR',
                        help='Directory that the "train" directory of DeepGlobe.')

    parser.add_argument('out_dir', metavar='OUTPUT_DIR',
                        help='Directory to save the crops to.')

    parser.add_argument('-n', type=int, default=3,
                        help='number of crops to take per image')

    parser.add_argument('--delete', action='store_true', help='Delete original files after.')

    args = parser.parse_args()
    main(args)

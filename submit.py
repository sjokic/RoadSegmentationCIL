""" Taking predicted masks, perform any post-processing to generate the kaggle submission. """

import os

import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm

from argparse import ArgumentParser
from utils.mask_to_submission import masks_to_submission
from utils.post_processing import pp_index


def post_process(pred, pp_steps, path_to_test_img, path_to_binary_mask):
    """
    Given an nparray of predictions, and a list of post_processing functions, apply the functions
    in the given order, convert to an image (mask) and write to the output path.
    """
    test_img = Image.open(path_to_test_img)
    test_img = np.asarray(test_img)

    # Apply post-processing steps
    for pp in pp_steps:
        pred = pp(pred, test_img)

    # Convert to image form
    mask = np.uint8(pred * 255)

    # Write mask as image
    mask_img = Image.fromarray(mask, 'L')
    mask_img.save(path_to_binary_mask)


def main(args):
    # Make directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Map test_file names to nparrays
    preds = {}

    # If weighted mean, search for the val_score.txt file of each dir
    weights = {pred_dir: 1. for pred_dir in args.pred_dirs}

    if args.weighted:
        for pred_dir in args.pred_dirs:
            val_score_f = os.path.join(pred_dir, 'weight.txt')

            if not os.path.exists(val_score_f):
                print('Weighted mean mode needs weight.txt files in each directory!')
                return

            with open(val_score_f, 'r') as f:
                val_score = float(f.readline())
                weights[pred_dir] = val_score

    total_weight = sum([weight for _, weight in weights.items()])

    # Get the list of post-processing steps
    pp_steps = [pp_index[pp] for pp in args.pp]

    # Select the first pred_dir and use that as our directory.
    # All the pred_dirs NEED the same files anyway.
    pred_files = os.listdir(args.pred_dirs[0])
    pred_files = [f for f in pred_files if f.endswith('.png')]

    print('Reading predictions.')
    for pred_file in tqdm(pred_files):
        # Go through each pred dir and sum the pred probs
        for pred_dir in args.pred_dirs:
            pred = mpimg.imread(os.path.join(pred_dir, pred_file))

            if pred_file not in preds:
                preds[pred_file] = pred * weights[pred_dir]
            else:
                preds[pred_file] += pred * weights[pred_dir]

    for pred_file in preds.keys():
        preds[pred_file] /= total_weight

    masks = []

    print('\nPost-processing predictions.')
    for pred_file, pred in tqdm(preds.items()):
        path_to_binary_mask = os.path.join(args.out_dir, pred_file)
        path_to_test_img = os.path.join(args.test_dir, pred_file)

        post_process(pred, pp_steps, path_to_test_img, path_to_binary_mask)
        masks.append(path_to_binary_mask)

    # Generate submission file
    masks_to_submission(os.path.join(args.out_dir, 'submission.csv'), masks)

    # Print the kaggle
    print('\nkaggle competitions submit -c cil-road-segmentation-2021 -f {} -m "message"'.format(
        os.path.join(args.out_dir, 'submission.csv')))


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Create a Kaggle submission file using directory of predictions.')

    parser.add_argument('out_dir', metavar='OUTPUT_DIR',
                        help='Directory to output post-processed masks and the submission.csv file.')

    parser.add_argument('test_dir', metavar='TEST_DIR',
                        help='Directory that contains the test images')

    parser.add_argument('--pred_dirs', nargs='+', required=True,
                        help='List all pred directories to ensemble together, delimited by a space.')

    parser.add_argument('--weighted', action='store_true',
                        help='Compute ensemble using weighted mean. \
                              Requires a weight.txt in the directory containing the predictions which specifies the value for the weight.')

    parser.add_argument('--pp', nargs='+',
                        default=['binary'],
                        choices=list(pp_index.keys()),
                        help='Ordered list of post processing steps, space delimited.')

    args = parser.parse_args()
    main(args)

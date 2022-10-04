"""
from https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_3.ipynb
"""

import os
import torch
import numpy as np
import re
from glob import glob
from tqdm import tqdm
from .dataloader import np_to_tensor, load_all_from_path, image_to_patches

PATCH_SIZE = 16  # pixels per side of square patches
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing road

def create_submission(test_pred, test_filenames, submission_filename):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*PATCH_SIZE, i*PATCH_SIZE, int(patch_array[i, j])))

def patch_predict(model, test_path, out_path, gpu):
    device = 'cuda' if (torch.cuda.is_available() and gpu) else 'cpu'
    test_filenames = sorted(glob(test_path + '/*.png'))
    test_images = load_all_from_path(test_path)
    test_patches = np.moveaxis(image_to_patches(test_images), -1, 1)  # HWC to CHW
    test_patches = np.reshape(test_patches, (38, -1, 3, PATCH_SIZE, PATCH_SIZE))  # split in batches for memory constraints
    test_pred = [torch.sigmoid(model(np_to_tensor(batch, device))).detach().cpu().numpy() for batch in tqdm(test_patches)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.round(test_pred.reshape(test_images.shape[0], test_images.shape[1] // PATCH_SIZE, test_images.shape[1] // PATCH_SIZE))

    create_submission(test_pred, test_filenames, submission_filename=os.path.join(out_path, 'submission.csv'))

"""
Define custom post-processing functions here.

All functions should take 2 nparrays (preds, test_img) and return the new preds
"""

import numpy as np
import denseCRF
from PIL import Image
from skimage.morphology import skeletonize, opening, erosion


def threshold_binary(preds, test_img, thres=0.5):
    """
    Given an nparray mask with values in the range [0., 1.], apply a threshold to generate
    a binary nparray mask in the range {0, 1}.
    """
    return np.uint8(preds > thres)


def crf(preds, test_img):
    I = test_img
    Iq = np.asarray(I)

    # load initial labels, and convert it into an array 'prob' with shape [H, W, C]
    # where C is the number of labels
    # prob[h, w, c] means the probability of pixel at (h, w) belonging to class c.
    Lq = preds
    H, W = Lq.shape
    # prob = Lq[:, :, :2]
    # prob[:, :, 0] = 1.0 - prob[:, :, 0]
    prob = np.zeros((H, W, 2), dtype=np.float32)
    prob[:, :, 1] = Lq
    prob[:, :, 0] = 1.0 - prob[:, :, 1]

    w1 = 2.0  # weight of bilateral term
    alpha = 80    # spatial std
    beta = 13    # rgb  std
    w2 = 10.0   # weight of spatial term
    gamma = 3     # spatial std
    it = 5.0   # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    lab = denseCRF.densecrf(Iq, prob, param)

    return lab


def nothing(preds, test_img):
    """
    Quick test to see if the output should be thresholded or not.
    """
    return preds

def check(x,y,i,skeleton,width):
    H,W = skeleton.shape
    for x1 in range(max(x-width,0),min(x+width+1,W),1):
        for y1 in range(max(y-width,0),min(y+width+1,H),1):
            if skeleton[x1,y1] == i:
                return True

def generate_prob_layer(skeleton, width,step_size):
    H,W = skeleton.shape
    for i in np.arange(1,step_size,-step_size):
        for x in range(0,W):
            for y in range(0,H):
                if skeleton[x,y] == 0:
                    skeleton[x,y] = i-step_size if check(x,y,i,skeleton,width) else 0
    return (skeleton - np.full(skeleton.shape,0.5))

def prob_layer(preds, test_img):
    """
    produce a skeleton of the predictions, depending on this skeleton produce a probability layer that derives a higher probability for pixels near the skeleton to become road pixels
    add these probabilities to the existing predictions and use binary threshold to predict road pixels
    """
    final_threshold = 0.9  #threshold at the end to decide between road and non-road pixels
    width = 1 #number of pixels next to each other that should have the same probability
    step_size = 0.1 #decrease in probability by pixel_distance

    skeleton = crf(preds, test_img)
    skeleton = skeletonize(skeleton)
    prob_layer = generate_prob_layer(skeleton, width,step_size)
    result = preds + prob_layer
    result = threshold_binary(result,test_img, thres =final_threshold)
    result = erosion(result)
    return opening(result)


# lmao
pp_index = {
    'nothing': nothing,
    'binary': threshold_binary,
    'crf': crf,
    'problayer': prob_layer
}

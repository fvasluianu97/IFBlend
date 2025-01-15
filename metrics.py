import numpy as np
from math import log10


def mse(img1, img2):
    mse = np.mean(np.square(img1 - img2))
    return mse


def psnr(img1, img2):
    rmse = np.sqrt(mse(img1, img2))
    psnr = 20.0 * log10(255/rmse)
    return psnr

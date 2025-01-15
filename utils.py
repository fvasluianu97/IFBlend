from collections import OrderedDict

import cv2 as cv
import math
import numpy as np
import pytorch_ssim
import torch
import torch.nn.functional as F
from metrics import mse, psnr
from PIL import Image
from skimage.io import imsave
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from pytorch_msssim import ssim


def PRIm(x, level):
    b, c, h, w = x.shape
    osz = (h // level, w // level)
    return F.interpolate(x, size=osz, mode="bicubic")


def cv2pil(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    return img_pil


def shuffle_down(x, factor):

    b, c, h, w = x.shape

    assert h % factor == 0 and w % factor == 0, "H and W must be a multiple of " + str(factor) + "!"

    n = x.reshape(b, c, int(h/factor), factor, int(w/factor), factor)
    n = n.permute(0, 3, 5, 1, 2, 4)
    n = n.reshape(b, c*factor**2, int(h/factor), int(w/factor))

    return n


def shuffle_up(x, factor):
    b, c, h, w = x.shape

    assert c % factor**2 == 0, "C must be a multiple of " + str(factor**2) + "!"

    n = x.reshape(b, factor, factor, int(c/(factor**2)), h, w)
    n = n.permute(0, 3, 4, 1, 5, 2)
    n = n.reshape(b, int(c/(factor**2)), factor*h, factor*w)

    return n


def tensor_to_img(img_tensor):
    img_array = np.moveaxis(255 * img_tensor.cpu().detach().numpy(), 0, -1)
    img_array = np.rint(np.clip(img_array, 0, 255)).astype(np.dtype('uint8'))
    return img_array


def rgb2gray(image):
    rgb_image = 255 * image
    return 0.299 * rgb_image[0, :, :] + 0.587 * rgb_image[1, :, :] + 0.114 * rgb_image[2, :, :]


def compute_maxchann_map(img_flare, img_free):
    b, c, h, w = img_flare.shape
    maps = []
    for i in range(b):
        img_diff = torch.abs(img_flare[i, :, :, :] - img_free[i, :, :, :])
        img_map = torch.max(img_diff, 0).values
        img_map = (img_map - img_map.min()) / (img_map.max() - img_map.min())
        maps.append(img_map.unsqueeze(0).unsqueeze(0))
    batch_map = torch.cat(maps, dim=0)
    return batch_map


def normalize_weights_map(wmap):
    wmap = wmap.detach()
    b, c, h, w = wmap.shape
    for i in range(b):
        for j in range(c):
            m = wmap[i, j, :, :].min()
            M = wmap[i, j, :, :].max()
            wmap[i, j, :, :] = (wmap[i, j, :, :] - m) / (M - m)

    return wmap


def save_checkpoint(ckp_path, model, optimizer, scheduler):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, "{}/checkpoint.pt".format(ckp_path))

def load_checkpoint(ckp_path, model, optimizer, scheduler):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler


def validate_model(model_net, val_dataloader, save_disk=False, out_dir=None, lpips=None):
    model_net.eval()
    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    num_samples = len(val_dataloader)

    with torch.no_grad():
        smse = 0
        spsnr = 0
        sssim = 0
        slpips = 0

        for i, batch in enumerate(val_dataloader):
            input_img = Variable(batch[0].type(Tensor))
            gt_img = Variable(batch[1].type(Tensor))

            out_img = model_net(input_img)
            # out_img = input_img

            out_symm = 2 * (torch.clamp(out_img, 0, 1) - 0.5)
            gt_symm = 2 * (gt_img - 0.5)

            if lpips is not None:
                lpips_d = lpips.forward(out_symm, gt_symm).item()
            else:
                lpips_d = 0
            slpips += lpips_d

            # sssim += pytorch_ssim.ssim(gt_img.detach(), out_img.detach())
            sssim += ssim(gt_img.detach(), out_img.detach(), data_range=1, size_average=True)

            inp_img = tensor_to_img(input_img.detach().squeeze(0))
            gt_img = tensor_to_img(gt_img.detach().squeeze(0))
            out_img = tensor_to_img(out_img.detach().squeeze(0))

            if save_disk:
                imsave("{}/{}_in.png".format(out_dir, i), inp_img)
                imsave("{}/{}_out.png".format(out_dir, i), out_img)
                imsave("{}/{}_gt.png".format(out_dir, i), gt_img)

            smse += mse(gt_img, out_img)
            spsnr += sk_psnr(gt_img, out_img)

    val_report = {
        "MSE": smse / num_samples,
        "PSNR": spsnr / num_samples,
        "SSIM": sssim / num_samples,
        "LPIPS": slpips / num_samples
    }

    model_net.train()
    return val_report

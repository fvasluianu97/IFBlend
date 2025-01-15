import numpy as np
import torch
from pytorch_msssim import ssim
import torch.nn.functional as F
import wandb


def get_image_gradients(x):
    dx = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

    dx = dx.view((1, 1, 3, 3))

    dy = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

    dy = dy.view((1, 1, 3, 3))

    b, c, h, w = x.shape
    G_x = torch.zeros_like(x)
    G_y = torch.zeros_like(x)

    if torch.cuda.is_available():
        x = x.cuda()
        dx = dx.cuda()
        dy = dy.cuda()
        G_x = G_x.cuda()
        G_y = G_y.cuda()

    for k in range(b):
        for i in range(c):
            G_x[k, i, :, :] = F.conv2d(x[k, i, :, :].unsqueeze(0), dx, padding=1)
            G_y[k, i, :, :] = F.conv2d(x[k, i, :, :].unsqueeze(0), dy, padding=1)

    return G_x, G_y


def compute_gradient_loss(out, gt):
    dout_dx, dout_dy = get_image_gradients(out)
    dgt_dx, dgt_dy = get_image_gradients(gt)
    return F.l1_loss(dout_dx, dgt_dx) + F.l1_loss(dout_dy, dgt_dy)


def compute_ssim_loss(out, gt):
    return 1 - ssim(out, gt, data_range=1, size_average=True)


def compute_loss(out, gt, opt, mode='l1', field_loss_module=None):
    if mode == 'l1':
        reconstruction_loss = F.l1_loss(out, gt)
    else:
        reconstruction_loss = F.mse_loss(out, gt)

    if opt.alpha_1 > 0:
        ssim_loss = compute_ssim_loss(out, gt)
    else:
        ssim_loss = 0

    if opt.alpha_2 > 0:
        grad_loss = compute_gradient_loss(out, gt)
    else:
        grad_loss = 0

    content_loss = field_loss_module.compute_content_loss(out, gt)
    wandb.log({
		"PIX loss": reconstruction_loss,
		"CNT loss": content_loss,
		"SSIM loss": ssim_loss,
		"GRAD loss": grad_loss
		
	})
    
    return reconstruction_loss + opt.alpha_1 * ssim_loss + opt.alpha_2 * grad_loss + opt.alpha_3 * content_loss


if __name__ == '__main__':
    out = torch.rand((5, 3, 256, 256))
    gt = torch.rand((5, 3, 256, 256))

    outx, outy = get_image_gradients(out)
    print(outx.shape, outy.shape)

import argparse
import os
import torch
import lpips
from dataloader import ImageSet, ISTDImageSet
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import validate_model, load_checkpoint, save_checkpoint
from utils_model import get_model

import wandb
wandb.init(project="IFBLEND_evals")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ifblend", help="Name of the tested model")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=150, help="epoch from which to start lr decay")
    parser.add_argument("--n_steps", type=int, default=2, help="number of step decays for the learning rate")

    parser.add_argument("--data_src",  default="./data/AMBIENT6K", help="Path for the dataset directory")
    parser.add_argument("--res_dir", default="./final-results", help="Path for temporary results dir")
    parser.add_argument("--ckp_dir", default="./checkpoints", help="Path for temporary checkpoints dir")


    parser.add_argument("--load_from", default="IFBlend_ambient6k", help="Experiment containing the loaded checkpoint")

    opt = parser.parse_args()
    print(opt)

    model_net = get_model(opt.model_name)
    if torch.cuda.device_count() >= 1:
        model_net = torch.nn.DataParallel(model_net)

    optimizer = torch.optim.Adam(model_net.parameters(), lr=0.0002)

    scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.6)
    loss_fn = lpips.LPIPS(net='alex')


    if opt.data_src.endswith("6K"):
        val_dataloader = DataLoader(
                ImageSet(opt.data_src, "Test", size=None, aug=False), batch_size=1,
                shuffle=False, num_workers=8)
    else:
        val_dataloader = DataLoader(ISTDImageSet(opt.data_src, 'test', size=None, aug=False), batch_size=1,
                                    shuffle=True, num_workers=8)


    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
        model_net = model_net.cuda()
        loss_fn = loss_fn.cuda()
    else:
        Tensor = torch.FloatTensor


    load_model_checkpoint = "{}/{}/best/checkpoint.pt".format(opt.ckp_dir, opt.load_from)
    model_net, _, _ = load_checkpoint(load_model_checkpoint, model_net, optimizer, scheduler)

    out_path = "{}/{}/".format(opt.res_dir, opt.load_from)
    os.makedirs(out_path, exist_ok=True)
    val_report = validate_model(model_net, val_dataloader, save_disk=True, out_dir=out_path, lpips=loss_fn)

    print("Validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f} - LPIPS - {:.4f}".format(val_report["MSE"],
                                                                                          val_report["PSNR"],
                                                                                          val_report["SSIM"],
                                                                                          val_report["LPIPS"]))







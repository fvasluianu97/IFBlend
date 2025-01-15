import argparse
import os
import torch
from dataloader import ImageSet, ISTDImageSet
from loss import compute_loss
from perceptual_loss import PerceptualLossModule
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import validate_model, save_checkpoint, load_checkpoint
from utils_model import get_model

import wandb
wandb.init(project="IFBLEND_TRAIN")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ifblend", help="Name of the tested model")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=150, help="epoch from which to start lr decay")
    parser.add_argument("--n_steps", type=int, default=2, help="number of step decays for the learning rate")

    parser.add_argument("--data_src",  default="data/AMBIENT", help="Path for the dataset directory")
    parser.add_argument("--res_dir", default="./results", help="Path for temporary results dir")
    parser.add_argument("--ckp_dir", default="./checkpoints", help="Path for temporary checkpoints dir")

    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")

    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--clip", type=int, default=0, help="Gradient clipping")
    
    parser.add_argument("--alpha_1", type=float, default=0.7, help="weight for SSIM loss")
    parser.add_argument("--alpha_2", type=float, default=0.1, help="weight for gradient loss")
    parser.add_argument("--alpha_3", type=float, default=0.1, help="weight for content loss")

    parser.add_argument("--img_height", type=int, default=448, help="size of image height")
    parser.add_argument("--img_width", type=int, default=448, help="size of image width")
    parser.add_argument("--valid_checkpoint", type=int, default=1, help="checkpoint for visual inspection")
    parser.add_argument("--save_checkpoint", type=int, default=10, help="Save checkpoint interval")
    parser.add_argument("--description", default="IFBLEND_base", help="Experiment name")
    parser.add_argument("--load", type=int, default=0, help="Load the best checkpoint from disk")
    parser.add_argument("--load_from", default="", help="Experiment containing the loaded checkpoint")

    opt = parser.parse_args()
    print(opt)

    model_net = get_model(opt.model_name)
    if torch.cuda.device_count() >= 1:
        model_net = torch.nn.DataParallel(model_net)

    betas = (opt.b1, opt.b2)
    optimizer = torch.optim.Adam(model_net.parameters(), lr=opt.lr, betas=betas)

    step_sz = (opt.n_epochs - opt.decay_epoch) // opt.n_steps
    milestones = [opt.decay_epoch + k * step_sz for k in range(opt.n_steps)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.6)

    if opt.data_src.endswith("ASRD6K"):
        train_dataloader = DataLoader(
                ImageSet(opt.data_src, "Train", size=(opt.img_height, opt.img_width)), batch_size=opt.batch_size,
                shuffle=True, num_workers=opt.n_cpu)

        val_dataloader = DataLoader(
                ImageSet(opt.data_src, "Validation", aug=False), batch_size=1,
                shuffle=False, num_workers=opt.n_cpu)
    else:
        train_dataloader = DataLoader(ISTDImageSet(opt.data_src, 'train', size=(opt.img_height, opt.img_width), aug=True), batch_size=opt.batch_size,
                                      shuffle=True, num_workers=opt.n_cpu)

        val_dataloader = DataLoader(ISTDImageSet(opt.data_src, 'test', size=None, aug=False), batch_size=1,
                                    shuffle=True, num_workers=opt.n_cpu)


    if torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
        model_net = model_net.cuda()
        field_loss_module = PerceptualLossModule(device=torch.device("cuda"))
    else:
        Tensor = torch.FloatTensor
        field_loss_module = PerceptualLossModule(device=torch.device("cpu"))

    
    

    best_psnr = 0
    bst_ckp_path = "{}/{}/best".format(opt.ckp_dir, opt.description)
    os.makedirs(bst_ckp_path, exist_ok=True)

    if opt.load == 1:
        load_model_checkpoint = "{}/{}/best/checkpoint.pt".format(opt.ckp_dir, opt.load_from)
        model_net, _, _ = load_checkpoint(load_model_checkpoint, model_net, optimizer, scheduler)

        out_path = "{}/{}/best_start".format(opt.res_dir, opt.description)
        os.makedirs(out_path, exist_ok=True)
        val_report = validate_model(model_net, val_dataloader, save_disk=True, out_dir=out_path)

        wandb.log({
            "val_mse": val_report["MSE"],
            "val_psnr": val_report["PSNR"],
            "val_ssim": val_report["SSIM"],
        })

        best_psnr = val_report["PSNR"]
        print("Starting at - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(val_report["MSE"],
                                                                                          val_report["PSNR"],
                                                                                          val_report["SSIM"]))
    else:
        os.makedirs(bst_ckp_path, exist_ok=True)

    for epoch in range(opt.n_epochs):
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            input_img = Variable(batch[0].type(Tensor))
            gt_img = Variable(batch[1].type(Tensor))
            optimizer.zero_grad()
            out_img = model_net(input_img)
            loss = compute_loss(out_img, gt_img, opt, field_loss_module=field_loss_module)
            loss.backward()
            
            if opt.clip == 1:
                torch.nn.utils.clip_grad_norm_(model_net.parameters(), 0.01)
            
            optimizer.step()
            epoch_loss += loss.detach().item()

        scheduler.step()
        print("Epoch: {} - training_loss {:.4f}".format(epoch + 1, epoch_loss))

        if (epoch + 1) % opt.valid_checkpoint == 0:
            save_disk = (epoch + 1) % opt.save_checkpoint == 0

            if save_disk:
                out_path = "{}/{}/{}".format(opt.res_dir, opt.description, epoch + 1)
                ckp_path = "{}/{}/{}".format(opt.ckp_dir, opt.description, epoch + 1)

                os.makedirs(out_path, exist_ok=True)
                os.makedirs(ckp_path, exist_ok=True)
                save_checkpoint(ckp_path, model_net, optimizer, scheduler)
            else:
                out_path = None

            val_report = validate_model(model_net, val_dataloader, save_disk=save_disk, out_dir=out_path)
            torch.cuda.empty_cache()

            wandb.log({
                "val_mse": val_report["MSE"],
                "val_psnr": val_report["PSNR"],
                "val_ssim": val_report["SSIM"],
            })

            
            if val_report["PSNR"] > best_psnr:
                best_psnr = val_report["PSNR"]
                save_checkpoint(bst_ckp_path, model_net, optimizer, scheduler)

            print("Epoch: {} - validation MSE: {:.3f} - PSNR: {:.3f} - SSIM: {:.4f}".format(epoch + 1, val_report["MSE"],
                                                           val_report["PSNR"], val_report["SSIM"]))





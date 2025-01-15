import os
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from random import random, choice
from PIL import Image
from torchvision.transforms import InterpolationMode


class ImageSet(data.Dataset):
    def __init__(self, set_path, set_type, aug=True, size=(512, 512), mode='rcrop'):
        path_dir = '{}/{}'.format(set_path, set_type)

        self.mode = mode
        self.size = size
        self.aug = aug

        self.gt_list = []
        self.inp_list = []
        self.num_samples = 0

        file_list = [f for f in os.listdir(path_dir) if f.endswith("gt.png")]

        for f in file_list:
            inp_path = os.path.join(path_dir, f.replace("gt", "in"))
            gt_path = os.path.join(path_dir, f)

            self.inp_list.append(inp_path)
            self.gt_list.append(gt_path)
            self.num_samples += 1

        assert len(self.inp_list) == len(self.gt_list)
        assert len(self.inp_list) == self.num_samples

    def __len__(self):
        return self.num_samples

    def augs(self, inp, gt):
        if self.mode == 'rcrop':
            w, h = gt.size
            tl = np.random.randint(0, h - self.size[0])
            tt = np.random.randint(0, w - self.size[1])

            gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
            inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])
        else:
            gt = torchvision.transforms.functional.resize(gt, self.size, InterpolationMode.BICUBIC)
            inp = torchvision.transforms.functional.resize(inp, self.size, InterpolationMode.BICUBIC)

        if random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random() < 0.5:
            angle = choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return inp, gt

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_list[index])
        gt_data = Image.open(self.gt_list[index])

        to_tensor = transforms.ToTensor()

        if self.aug:
            inp_data, gt_data = self.augs(inp_data, gt_data)
        else:
            if self.size is not None:
                inp_data = torchvision.transforms.functional.resize(inp_data, self.size, InterpolationMode.BICUBIC)
                gt_data = torchvision.transforms.functional.resize(gt_data, self.size, InterpolationMode.BICUBIC)

        return to_tensor(inp_data), to_tensor(gt_data)


class ISTDImageSet(data.Dataset):
    def __init__(self, set_path, set_type, size=(256, 256), use_mask=True, aug=False):
        self.augment = aug

        self.size = size
        self.use_mask = use_mask

        self.to_tensor = transforms.ToTensor()
        if size is not None:
            self.resize = transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC)
        else:
            self.resize = None

        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)

        self.gt_images_path = []
        self.masks_path = []
        self.inp_images_path = []
        self.num_samples = 0

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                if f.endswith(".zip"):
                    continue
                orig_path = os.path.join(dirpath, f)
                clean_path = os.path.join(clean_path_dir, f)

                self.gt_images_path.append(clean_path)
                self.inp_images_path.append(orig_path)

                self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def augs(self, gt, inp):
        w, h = gt.size
        tl = np.random.randint(0, h - self.size[0])
        tt = np.random.randint(0, w - self.size[1])

        gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
        inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])

        if random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random() < 0.5:
            angle = choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return gt, inp

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_images_path[index])
        gt_data = Image.open(self.gt_images_path[index])

        if self.augment:
            gt_data, inp_data = self.augs(gt_data, inp_data)
        else:
            if self.resize is not None:
                gt_data = self.resize(gt_data)
                inp_data = self.resize(inp_data)

        tensor_gt = self.to_tensor(gt_data)
        tensor_inp = self.to_tensor(inp_data)

        return tensor_inp, tensor_gt



class ISTDImageMaskSet(data.Dataset):
    def __init__(self, set_path, set_type, size=(256, 256), use_mask=True, aug=False):
        self.augment = aug

        self.size = size
        self.use_mask = use_mask

        self.to_tensor = transforms.ToTensor()
        if size is not None:
            self.resize = transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC)
        else:
            self.resize = None

        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)
        mask_path_dir = '{}/{}/{}_B'.format(set_path, set_type, set_type)

        self.gt_images_path = []
        self.masks_path = []
        self.inp_images_path = []
        self.num_samples = 0

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                if f.endswith(".zip"):
                    continue
                orig_path = os.path.join(dirpath, f)
                clean_path = os.path.join(clean_path_dir, f)
                mask_path = os.path.join(mask_path_dir, f)

                self.gt_images_path.append(clean_path)
                self.inp_images_path.append(orig_path)
                self.masks_path.append(mask_path)

                self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def augs(self, gt, inp):
        w, h = gt.size
        tl = np.random.randint(0, h - self.size[0])
        tt = np.random.randint(0, w - self.size[1])

        gt = torchvision.transforms.functional.crop(gt, tt, tl, self.size[0], self.size[1])
        inp = torchvision.transforms.functional.crop(inp, tt, tl, self.size[0], self.size[1])

        if random() < 0.5:
            inp = torchvision.transforms.functional.hflip(inp)
            gt = torchvision.transforms.functional.hflip(gt)
        if random() < 0.5:
            inp = torchvision.transforms.functional.vflip(inp)
            gt = torchvision.transforms.functional.vflip(gt)
        if random() < 0.5:
            angle = choice([90, 180, 270])
            inp = torchvision.transforms.functional.rotate(inp, angle)
            gt = torchvision.transforms.functional.rotate(gt, angle)

        return gt, inp

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_images_path[index])
        gt_data = Image.open(self.gt_images_path[index])
        smat_data = Image.open(self.masks_path[index])

        if self.resize is not None:
            gt_data = self.resize(gt_data)
            inp_data = self.resize(inp_data)
            smat_data = self.resize(smat_data)


        tensor_gt = self.to_tensor(gt_data)
        tensor_mask = self.to_tensor(smat_data)
        tensor_inp = self.to_tensor(inp_data)

        return tensor_inp, tensor_mask, tensor_gt

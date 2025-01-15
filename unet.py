import torch.nn as nn
import torch.nn.functional as F
import torch


class UNetCompress(nn.Module):
    def __init__(self, in_size, out_size, normalize=True,  kernel_size=4, dropout=0.33):
        super(UNetCompress, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetDecompress(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.13):
        super(UNetDecompress, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        f = self.model(x)
        return f

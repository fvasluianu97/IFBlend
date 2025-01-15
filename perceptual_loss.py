import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16

os.environ['TORCH_HOME'] = "./loaded_models/"


def gram_matrix(i_input):
    a, b, c, d = i_input.size()
    features = i_input.view(a * b, c * d)
    Gm = torch.mm(features, features.t())
    return Gm.div(a * b * c * d)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in {3, 8, 15, 22}:
                results.append(x)
        return results


class PerceptualLossModule:
    def __init__(self, device):
        self.device = device
        self.model = Vgg16()
        self.criterion = torch.nn.MSELoss()
        self.model.to(self.device)
        self.criterion.to(self.device)

    def compute_content_loss(self, i_input, target):
        input_feats = self.model(i_input)
        target_feats = self.model(target)
        nr_feats = len(input_feats)
        content_loss = 0
        for i in range(nr_feats):
            content_loss += self.criterion(input_feats[i], target_feats[i]).item()
        return content_loss / nr_feats

# Implementation from https://github.com/hzwer/Practical-RIFE/blob/main/model/loss.py

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True
        ).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i + 1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss


class MeanShift(nn.Conv2d):
    """Use Conv2d to perform mean shift operation."""

    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(0, 1, (3, 3, 256, 256))).float().to(device)
    vgg_perceptual_loss = VGGPerceptualLoss().to(device)
    print(vgg_perceptual_loss(img0, img1))

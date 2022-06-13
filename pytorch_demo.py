import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


print("Testing stuff")


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.n_filters = 16

        self.res_block == nn.Sequential(
            nn.Conv2d(self.n_filters, self.n_filters, 3, padding='zeros', stride=(1, 1)),
            nn.BatchNorm2d(),
            nn.ReLU(),

            nn.Conv2d(self.n_filters, self.n_filters, padding='zeros', stride=(1, 1))
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

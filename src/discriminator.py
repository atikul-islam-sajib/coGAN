import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class CoupledDiscriminators(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        image_size: int = 32,
        constant: int = 128,
    ):
        super(CoupledDiscriminators, self).__init__()

        self.in_channels = channels
        self.image_size = image_size
        self.constant = constant

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        if isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor):
            pass

        else:
            raise ValueError("Both inputs must be PyTorch tensors".capitalize())


if __name__ == "__main__":
    pass

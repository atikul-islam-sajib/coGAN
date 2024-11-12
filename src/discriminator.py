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
        self.out_channels = self.in_channels * 5 + 1
        self.image_size = image_size
        self.constant = constant

        self.kernel_size = 3
        self.stride_size = 2
        self.padding_size = 1

        self.layers = []

        for index in range(4):
            self.layers += [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                ),
            ]

            if index != 0:
                self.layers += [nn.BatchNorm2d(num_features=self.out_channels)]

            self.layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]

            self.in_channels = self.out_channels
            self.out_channels = self.in_channels * 2

        self.sharedConvolution = nn.Sequential(*self.layers)

        self.discriminator1 = nn.Linear(
            in_features=self.constant * (self.image_size // 2**4) ** 2,
            out_features=self.in_channels // self.in_channels,
        )
        
        self.discriminator2 = nn.Linear(
            in_features=self.constant * (self.image_size // 2**4) ** 2,
            out_features=self.in_channels // self.in_channels,
        )

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        if isinstance(image1, torch.Tensor) and isinstance(image2, torch.Tensor):
            shared = self.sharedConvolution(image1)

            shared = shared.view(shared.size(0), -1)
            
            validity1 = self.discriminator1(shared)
            validity2 = self.discriminator2(shared)
            
            return validity1, validity2

        else:
            raise ValueError("Both inputs must be PyTorch tensors".capitalize())


if __name__ == "__main__":
    netD = CoupledDiscriminators()
    validity1, validity2 = netD(
            image1=torch.randn((64, 3, 32, 32)), image2=torch.randn((64, 3, 32, 32))
        )
    
    
    print("Validity1 size # {}".format(validity1.size()))
    print("Validity2 size # {}".format(validity2.size()))

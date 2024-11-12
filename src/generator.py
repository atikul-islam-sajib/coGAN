import sys
import os
import argparse
import torch
import torch.nn as nn


class CoupledGenerators(nn.Module):
    def __init__(
        self,
        latent_space: int = 100,
        constant: int = 128,
        image_size: int = 32,
    ):
        super(CoupledGenerators, self).__init__()
        self.latent_space = latent_space
        self.constant = constant
        self.image_size = image_size

        self.kernel_size = 3
        self.stride_size = 1
        self.padding_size = 1

        self.negative_slope = 0.2
        self.scale_factor = 2

        self.fullyConnectedLayer = nn.Linear(
            in_features=self.latent_space,
            out_features=self.constant * self.image_size // 4 * self.image_size // 4,
        )

        self.sharedConvolution = nn.Sequential(
            nn.BatchNorm2d(self.constant),
            nn.Upsample(scale_factor=self.scale_factor),
            nn.Conv2d(
                in_channels=self.constant,
                out_channels=self.constant,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
            ),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.BatchNorm2d(self.constant),
            nn.Upsample(scale_factor=self.scale_factor),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self.fullyConnectedLayer(x)
            x = x.view(
                x.size(0), self.constant, self.image_size // 4, self.image_size // 4
            )

            return self.sharedConvolution(x)

        else:
            raise ValueError("Input should be a torch.Tensor".capitalize())


if __name__ == "__main__":
    netG = CoupledGenerators()
    print(netG(torch.randn(64, 100)).size())

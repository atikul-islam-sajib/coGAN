import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

from utils import config


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

        self.netG1Layers = []
        self.netG2Layers = []

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

        for index in range(2):
            self.netG1Layers.append(
                nn.Conv2d(
                    in_channels=self.constant if index == 0 else self.constant // 2,
                    out_channels=self.constant // 2 if index == 0 else 3,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                )
            )
            (
                self.netG1Layers.append(
                    nn.Sequential(
                        nn.BatchNorm2d(num_features=self.constant // 2),
                        nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
                    )
                )
                if index == 0
                else nn.Tanh()
            )

        for index in range(2):
            self.netG2Layers.append(
                nn.Conv2d(
                    in_channels=self.constant if index == 0 else self.constant // 2,
                    out_channels=self.constant // 2 if index == 0 else 3,
                    kernel_size=self.kernel_size,
                    stride=self.stride_size,
                    padding=self.padding_size,
                )
            )
            (
                self.netG2Layers.append(
                    nn.Sequential(
                        nn.BatchNorm2d(num_features=self.constant // 2),
                        nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
                    )
                )
                if index == 0
                else nn.Tanh()
            )

        self.generator1 = nn.Sequential(*self.netG1Layers)
        self.generator2 = nn.Sequential(*self.netG2Layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self.fullyConnectedLayer(x)
            x = x.view(
                x.size(0), self.constant, self.image_size // 4, self.image_size // 4
            )

            shared = self.sharedConvolution(x)

            image1 = self.generator1(shared)
            image2 = self.generator2(shared)

            return image1, image2

        else:
            raise ValueError("Input should be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CoGAN generator for Creating Images".title()
    )
    parser.add_argument(
        "--latent_space",
        default=config()["netG"]["latent_space"],
        choices=[100, 200, 300],
        type=int,
        help="Dimensionality of the latent space".capitalize(),
    )
    parser.add_argument(
        "--constant",
        default=config()["netG"]["constant"],
        choices=[64, 128, 256],
        type=int,
        help="The constant to use for the latent space".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        default=config()["dataloader"]["image_size"],
        choices=[32, 64, 128],
        type=int,
    )

    args = parser.parse_args()

    latent_space = args.latent_space
    constant = args.constant
    image_size = args.image_size

    batch_size = config()["dataloader"]["batch_size"]

    netG = CoupledGenerators(
        latent_space=latent_space,
        constant=constant,
        image_size=image_size,
    )

    image1, image2 = netG(torch.randn(batch_size, latent_space))

    assert (
        image1.size() == image2.size()
    ), "Image1 and Image2 must be the same size".capitalize()

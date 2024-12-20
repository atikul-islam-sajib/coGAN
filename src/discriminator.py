import os
import sys
import torch
import argparse
import torch.nn as nn
from torchview import draw_graph

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
            shared1 = self.sharedConvolution(image1)
            shared1 = shared1.view(shared1.size(0), -1)

            shared2 = self.sharedConvolution(image2)
            shared2 = shared2.view(shared2.size(0), -1)

            validity1 = self.discriminator1(shared1)
            validity2 = self.discriminator2(shared2)

            return validity1, validity2

        else:
            raise ValueError("Both inputs must be PyTorch tensors".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discriminator for the coupleGAN".title()
    )
    parser.add_argument(
        "--image_size",
        default=config()["dataloader"]["image_size"],
        type=int,
        help="Th size of the image to be loaded".capitalize(),
    )
    parser.add_argument(
        "--constant",
        default=config()["netG"]["constant"],
        choices=[
            128,
        ],
        type=int,
        help="The constant to use for the latent space".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["dataloader"]["batch_size"]
    image_size = args.image_size
    constant = args.constant
    channels = 3

    netD = CoupledDiscriminators(
        channels=channels,
        image_size=image_size,
        constant=constant,
    )

    validity1, validity2 = netD(
        image1=torch.randn((batch_size, channels, image_size, image_size)),
        image2=torch.randn((batch_size, channels, image_size, image_size)),
    )

    assert (
        validity1.size() == validity2.size()
    ), "Validity1 and Validity2 must be the same size".capitalize()

    print("Validity1 size # {}".format(validity1.size()))
    print("Validity2 size # {}".format(validity2.size()))

    try:
        input_data1 = torch.randn((batch_size, channels, image_size, image_size))
        input_data2 = torch.randn((batch_size, channels, image_size, image_size))

        draw_graph(
            model=netD, input_data=(input_data1, input_data2)
        ).visual_graph.render(
            filename=os.path.join("./artifacts/files/", "coupleDiscriminator"),
            format="png",
        )
        print("Graph saved in ./artifacts/files/coupleDiscriminator.png")
    except Exception as e:
        print(f"Error during graph rendering: {e}")

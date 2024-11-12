import os
import sys
import torch
import torch.nn as nn
import unittest


sys.path.append("./src")

from utils import config
from generator import CoupledGenerators


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.image_size = config()["dataloader"]["image_size"]
        self.batch_size = config()["dataloader"]["batch_size"]

        self.latent_space = config()["netG"]["latent_space"]
        self.constant = config()["netG"]["constant"]

        self.netG = CoupledGenerators(
            latent_space=self.latent_space,
            constant=self.constant,
            image_size=self.image_size,
        )

    def test_coupleGenerator(self):
        self.Z = torch.randn(self.batch_size, self.latent_space)

        image1, image2 = self.netG(self.Z)

        self.assertEqual(
            image1.size(), image2.size(), "Image1 and Image2 must be the same size"
        )


if __name__ == "__main__":

    unittest.main()

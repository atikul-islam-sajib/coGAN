import os
import sys
import torch
import unittest
import torch.nn as nn
import torch.optim as optim


sys.path.append("./src")

from utils import config, load
from loss import AdversarialLoss
from generator import CoupledGenerators
from discriminator import CoupledDiscriminators
from dataloader import Loader
from helper import helper


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.image_size = config()["dataloader"]["image_size"]
        self.batch_size = config()["dataloader"]["batch_size"]
        self.split_size = config()["dataloader"]["split_size"]

        self.latent_space = config()["netG"]["latent_space"]
        self.constant = config()["netG"]["constant"]

        self.dataset = config()["dataloader"]["dataset"]

        self.adam = config()["trainer"]["adam"]
        self.SGD = config()["trainer"]["SGD"]
        self.beta1 = float(config()["trainer"]["beta1"])
        self.beta2 = float(config()["trainer"]["beta2"])
        self.lr = float(config()["trainer"]["lr"])
        self.momentum = float(config()["trainer"]["momentum"])
        self.regularizer = float(config()["trainer"]["regularizer"])
        self.device = config()["trainer"]["device"]

        self.loader = Loader(
            dataset=self.dataset,
            image_size=self.image_size,
            batch_size=self.batch_size,
            split_size=self.split_size,
        )

        self.netG = CoupledGenerators(
            latent_space=self.latent_space,
            constant=self.constant,
            image_size=self.image_size,
        )

        self.netD = CoupledDiscriminators(
            channels=3,
            constant=self.constant,
            image_size=self.image_size,
        )

        self.adversarial_loss = AdversarialLoss()

        self.init = helper(
            adam=self.adam,
            SGD=self.SGD,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            momentum=self.momentum,
            reduction="mean",
        )

    def test_coupleGenerator(self):
        self.Z = torch.randn(self.batch_size, self.latent_space)

        image1, image2 = self.netG(self.Z)

        self.assertEqual(
            image1.size(), image2.size(), "Image1 and Image2 must be the same size"
        )

    def test_coupledDiscriminator(self):
        self.Z = torch.randn(self.batch_size, self.latent_space)

        image1, image2 = self.netG(self.Z)

        validity1, validity2 = self.netD(image1=image1, image2=image2)

        self.assertEqual(
            validity1.size(),
            validity2.size(),
            "Validity1 and Validity2 must be the same size",
        )

    def test_adversarialLoss(self):
        predicted = torch.Tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        actual = torch.Tensor([1.0, 0.0, 1.0, 0.0, 0.0])

        self.assertIsInstance(
            self.adversarial_loss(pred=predicted, actual=actual), torch.Tensor
        )

    def test_dataloader(self):
        train_dataloader = os.path.join(
            config()["path"]["processed_path"], "train_dataloader.pkl"
        )
        valid_dataloader = os.path.join(
            config()["path"]["processed_path"], "valid_dataloader.pkl"
        )

        train_dataloader = load(filename=train_dataloader)
        valid_dataloader = load(filename=valid_dataloader)

        X11, X12 = next(iter(train_dataloader))
        X21, X22 = next(iter(valid_dataloader))

        self.assertEqual(
            X11.size(),
            X12.size(),
            "Image1 and Image2 from training dataloader must be the same size".capitalize(),
        )

        self.assertEqual(
            X21.size(),
            X22.size(),
            "Image1 and Image2 from validation dataloader must be the same size".capitalize(),
        )

        self.assertEqual(
            X11.size(0),
            config()["dataloader"]["batch_size"],
            "batch_size should be matched",
        )
        self.assertEqual(
            X21.size(2),
            config()["dataloader"]["image_size"],
            "image_size should be matched",
        )

    def test_helper_method(self):
        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.netG = self.init["netG"]
        self.netD = self.init["netD"]

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD = self.init["optimizerD"]

        self.adversarial_loss = self.init["adversarial_loss"]

        assert (
            self.train_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Train Dataloader should be the class of dataloader".capitalize()
        assert (
            self.valid_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Validation Dataloader should be the class of dataloader".capitalize()

        self.netG.__class__ == CoupledGenerators, "Generator should be the class of CoupledGenerators".capitalize()
        self.netD.__class__ == CoupledDiscriminators, "Discriminator should be the class of CoupledDiscriminators".capitalize()

        if self.adam:
            assert (
                self.optimizerG.__class__ == optim.Adam
            ), "OptimizerG should be the class of Adam".capitalize()
            assert (
                self.optimizerD.__class__ == optim.Adam
            ), "OptimizerD should be the class of Adam".capitalize()
        else:
            assert (
                self.optimizerG.__class__ == optim.SGD
            ), "OptimizerG should be the class of SGD".capitalize()
            assert (
                self.optimizerD.__class__ == optim.SGD
            ), "OptimizerD should be the class of SGD".capitalize()

        self.adversarial_loss.__class__ == AdversarialLoss, "AdversarialLoss should be the class of Adversarial".capitalize()


if __name__ == "__main__":
    unittest.main()

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append("./src/")

from utils import config, load, device_init
from helper import helper
from loss import AdversarialLoss
from generator import CoupledGenerators
from discriminator import CoupledDiscriminators


class Trainer:
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 2e-4,
        momentum: float = 0.75,
        beta1: float = 0.5,
        beta2: float = 0.999,
        device: str = "cuda",
        adam: bool = True,
        SGD: bool = False,
        l1_regularization: bool = False,
        l2_regularization: bool = False,
        elasticnet_regularization: bool = False,
        mlflow: bool = False,
        verbose: bool = True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticnet_regularization = elasticnet_regularization
        self.mlflow = mlflow
        self.verbose = verbose

        self.init = helper(
            adam=self.adam,
            SGD=self.SGD,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            momentum=self.momentum,
            reduction="mean",
        )

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

        self.device = device_init(device=self.device)

    def saved_checkpoints(self, **kwargs):
        pass

    def update_netG_training(self, **kwargs):
        self.optimizerG.zero_grad()

        image1 = kwargs["image1"]
        image2 = kwargs["image2"]

        latent_dim = kwargs["latent_dim"]

        generated_image1, generated_image2 = self.netG(latent_dim)

        predicted_image1, predicted_image2 = self.netD(
            generated_image1, generated_image2
        )

        predicted_image1_loss = self.adversarial_loss(
            predicted_image1, torch.ones_like(predicted_image1)
        )
        predicted_image2_loss = self.adversarial_loss(
            predicted_image2, torch.ones_like(predicted_image2)
        )

        total_loss = (predicted_image1_loss + predicted_image2_loss) / 2

        total_loss.backward()
        self.optimizerG.step()

        return total_loss.item()

    def update_netD_training(self, **kwargs):
        pass

    def display_progress(self, **kwargs):
        pass

    def train(self):
        for index, epoch in tqdm(range(self.epochs)):
            train_loss = []
            valid_loss = []
            for idx, (image1, image2) in enumerate(self.train_dataloader):
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)

                batch_size = image1.size(0)
                latent_dim = torch.randn(
                    (batch_size, config()["dataloader"]["latent_space"])
                ).to(self.device)

                train_loss.append(
                    self.update_netG_training(
                        image1=image1, image2=image2, latent_dim=latent_dim
                    )
                )

    @staticmethod
    def model_history():
        pass


if __name__ == "__main__":
    trainer = Trainer(
        epochs=1,
    )

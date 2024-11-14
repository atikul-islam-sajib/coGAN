import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils

sys.path.append("./src/")

from utils import config, load
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

    def saved_checkpoints(self, **kwargs):
        pass

    def update_netG_training(self, **kwargs):
        pass

    def update_netD_training(self, **kwargs):
        pass

    def display_progress(self, **kwargs):
        pass

    def train(self):
        pass

    @staticmethod
    def model_history():
        pass


if __name__ == "__main__":
    trainer = Trainer(
        epochs=1,
    )

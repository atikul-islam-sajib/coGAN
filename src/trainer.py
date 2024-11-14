import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")

from utils import config, load
from helper import helper


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
        
        
        self.init = pass

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

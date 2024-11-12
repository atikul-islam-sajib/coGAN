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
    ):
        super(CoupledGenerators, self).__init__()
        self.latent_space = latent_space
        self.constant = constant
        
    def forward(self, x):
        pass


if __name__ == "__main__":
    pass
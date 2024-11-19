import os
import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src/")

class GANLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(GANLoss, self).__init__()
        
        self.name = "GANLoss for the UNIT-GAN"
        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=self.reduction)
        
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor):
        if isinstance(predicted, torch.Tensor) and isinstance(actual, torch.Tensor):
            return self.loss(predicted, actual)
        else:
            raise ValueError("Predicted and actual should be both tensor".capitalize())
        
if __name__ == "__main__":
    loss = GANLoss(reduction="mean")
    actual = torch.tensor([1.0, 0.0, 1.0, 1.0])
    predicted = torch.tensor([1.0, 0.0, 1.0, 1.0])
    
    print(loss(predicted, actual))
import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


class AdversarialLoss(nn.Module):
    def __init__(
        self, name: str = "Adversarial Loss".capitalize(), reduction: str = "mean"
    ):
        super(AdversarialLoss, self).__init__()

        self.name = name
        self.reduction = reduction

        self.MSEloss = nn.MSELoss(reduction=self.reduction)

    def forward(self, pred: torch.Tensor, actual: torch.Tensor):
        if isinstance(pred, torch.Tensor) and isinstance(actual, torch.Tensor):
            loss = self.MSEloss(pred, actual)

            return loss
        else:
            raise ValueError(
                f"Both 'pred' and 'actual' should be torch.Tensor.".capitalize()
            )
            
            
if __name__ == "__main__":
    loss = AdversarialLoss()
    
    actual = torch.Tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    predicted = torch.Tensor([1.0, 0.0, 1.0, 1.0, 1.0])
    
    print(f"Adversarial Loss: {loss(predicted, actual):.4f}")

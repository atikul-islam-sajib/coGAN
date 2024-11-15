import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils

sys.path.append("./src/")

from utils import config, load
from loss import AdversarialLoss
from generator import CoupledGenerators
from discriminator import CoupledDiscriminators


def load_dataset():
    train_dataloader = os.path.join(
        config()["path"]["processed_path"], "train_dataloader.pkl"
    )
    valid_dataloader = os.path.join(
        config()["path"]["processed_path"], "valid_dataloader.pkl"
    )

    train_dataloader = load(filename=train_dataloader)
    valid_dataloader = load(filename=valid_dataloader)

    return {
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
    }


def helper(**kwargs):
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]

    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]

    lr = kwargs["lr"]
    momentum = kwargs["momentum"]

    reduction = kwargs["reduction"]

    netG = CoupledGenerators(
        latent_space=config()["netG"]["latent_space"],
        constant=config()["netG"]["constant"],
        image_size=config()["dataloader"]["image_size"],
    )

    netD = CoupledDiscriminators(
        channels=3,
        image_size=config()["dataloader"]["image_size"],
        constant=config()["netG"]["constant"],
    )

    if adam:
        optimizerG = optim.Adam(params=netG.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerD = optim.Adam(params=netD.parameters(), lr=lr, betas=(beta1, beta2))
        
    elif SGD:
        optimizerG = optim.SGD(params=netG.parameters(), lr=lr, momentum=momentum)
        optimizerD = optim.SGD(params=netD.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError("Invalid optimizer choice".capitalize())

    adversarial_loss = AdversarialLoss(
        name="Adversarial Loss for coupledGAN".title(), reduction=reduction
    )

    dataloader = load_dataset()

    return {
        "train_dataloader": dataloader["train_dataloader"],
        "valid_dataloader": dataloader["valid_dataloader"],
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "adversarial_loss": adversarial_loss,
    }


if __name__ == "__main__":
    init = helper(
        adam=False,
        SGD=True,
        beta1=0.5,
        beta2=0.999,
        lr=0.0002,
        momentum=0.0,
        reduction="mean",
    )

    assert init["train_dataloader"].__class__ == torch.utils.data.DataLoader
    assert init["valid_dataloader"].__class__ == torch.utils.data.DataLoader
    assert (
        init["netG"].__class__ == CoupledGenerators
    ), "netG should be coupledGenerators".title()
    assert (
        init["netD"].__class__ == CoupledDiscriminators
    ), "netD should be coupledDiscriminators".title()

    # assert init["optimizerG"].__class__ == optim.Adam, "optimizerG should be Adam".title()
    # assert init["optimizerD"].__class__ == optim.Adam, "optimizerD should be Adam".title()

    assert init["optimizerG"].__class__ == optim.SGD, "optimizerG should be SGD".title()
    assert init["optimizerD"].__class__ == optim.SGD, "optimizerD should be SGD".title()

    assert (
        init["adversarial_loss"].__class__ == AdversarialLoss
    ), "adversarial_loss should be AdversarialLoss".title()

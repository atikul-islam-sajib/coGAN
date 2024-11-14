import os
import sys
import yaml
import torch
import joblib
import torch.nn as nn

sys.path.append("./src/")


def dump(value: str, filename: str):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)
    else:
        raise ValueError(
            f"Both 'value' and 'filename' should be provided.".capitalize()
        )


def load(filename: str):
    if filename is not None:
        return joblib.load(filename=filename)
    else:
        raise ValueError(f"Please provide a valid 'filename' to load.".capitalize())


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def device_init(device: str = "cuda"):
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")

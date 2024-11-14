import os
import sys
import zipfile
import torch
import torch.nn as nn

sys.path.append("./src/")

from utils import config


class Loader:
    def __init__(self, dataset: str, image_size = 32, batch_size: int = 8, split_size: float = 0.25):
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

    def transforms(self):
        pass

    def split_dataset(self):
        pass

    def unzip_folder(self):
        if os.path.exists(path=config()["path"]["processed_path"]):
            with zipfile.ZipFile(file=self.dataset, mode="r") as zip_file:
                zip_file.extractall(path=config()["path"]["processed_path"])

        else:
            raise FileNotFoundError("File not found - processed data path".capitalize())

    def create_dataloader(self):
        pass

    @staticmethod
    def dataset_details():
        pass


if __name__ == "__main__":
    loader = Loader(
        dataset="/Users/macbook/Desktop/coGAN/data/raw/dataset.zip",
        batch_size=config()["dataloader"]["batch_size"],
        image_size=config()["dataloader"]["image_size"],
        split_size=config()["dataloader"]["split_size"],
    )
    
    loader.unzip_folder()

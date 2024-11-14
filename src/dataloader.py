import os
import sys
import torch
import torch.nn as nn

sys.path.append("./src/")


class Loader:
    def __init__(self, dataset: str, batch_size: int = 8, split_size: float = 0.25):
        self.dataset = dataset
        self.batch_size = batch_size
        self.split_size = split_size
        
    def transforms(self):
        pass
    
    def split_dataset(self):
        pass
    
    def unzip_dataset(self):
        pass
    
    def create_dataloader(self):
        pass
    
    @staticmethod
    def dataset_details():
        pass


if __name__ == "__main__":
    pass

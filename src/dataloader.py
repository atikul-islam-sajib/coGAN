import os
import sys
import cv2
import math
import zipfile
import argparse
import warnings
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("./src/")

from utils import config, dump, load

warnings.filterwarnings("ignore")


class Loader:
    def __init__(
        self, dataset: str, image_size=32, batch_size: int = 8, split_size: float = 0.25
    ):
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.X1 = []
        self.X2 = []

    def transforms(self, type: str = "coupled"):
        if type != "coupled":
            return transforms.Compose(
                [
                    transforms.Resize(size=(self.image_size, self.image_size)),
                    transforms.CenterCrop(size=(self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        else:
            return transforms.Compose(
                [
                    transforms.Resize(size=(self.image_size, self.image_size)),
                    transforms.CenterCrop(size=(self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=45),
                ]
            )

    def split_dataset(self, **dataset):
        X = dataset["X"]
        y = dataset["y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.split_size, random_state=42
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def features_extractor(self):
        dataset = os.path.join(config()["path"]["processed_path"], "dataset")

        for image in tqdm(os.listdir(dataset)):
            image = os.path.join(dataset, image)

            if (image is not None) and (image.endswith((".png", ".jpg", ".jpeg"))):
                image = cv2.imread(filename=image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = Image.fromarray(image)

                self.X1.append(self.transforms(type=None)(image))
                self.X2.append(self.transforms(type="coupled")(image))

            else:
                print(f"Invalid image: {image}".capitalize())

        assert len(self.X1) == len(
            self.X2
        ), "Length should be same while extracting the image".capitalize()

        dataset = self.split_dataset(X=self.X1, y=self.X2)

        return dataset

    def unzip_folder(self):
        if os.path.exists(path=config()["path"]["processed_path"]):
            with zipfile.ZipFile(file=self.dataset, mode="r") as zip_file:
                zip_file.extractall(path=config()["path"]["processed_path"])

        else:
            raise FileNotFoundError("File not found - processed data path".capitalize())

    def create_dataloader(self):
        try:
            dataset = self.features_extractor()

            train_dataloader = DataLoader(
                dataset=list(zip(dataset["X_train"], dataset["y_train"])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            valid_dataloader = DataLoader(
                dataset=list(zip(dataset["X_test"], dataset["y_test"])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            for filename, value in [
                ("train_dataloader", train_dataloader),
                ("valid_dataloader", valid_dataloader),
            ]:
                dump(
                    value=value,
                    filename=os.path.join(
                        config()["path"]["processed_path"], filename + ".pkl"
                    ),
                )

            print(
                "Dataloader is stored in the directory of {}".capitalize().format(
                    config()["path"]["processed_path"]
                )
            )

        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def display_images(self):
        dataloader = os.path.join(
            config()["path"]["processed_path"], "valid_dataloader.pkl"
        )

        dataloader = load(filename=dataloader)

        X1, X2 = next(iter(dataloader))

        assert (
            X1.size() == X2.size()
        ), "Cannot be possible to display the images in the same order".capitalize()

        num_of_rows = int(math.sqrt(X1.size(0)))
        num_of_columns = X1.size(0) // num_of_rows

        plt.figure(figsize=(40, 15))

        plt.suptitle("Training Images".title())

        for index, image1 in enumerate(X1):
            image1 = image1.squeeze().permute(1, 2, 0).numpy()
            image1 = (image1 - image1.min()) / (image1.max() - image1.min())

            image2 = X2[index].squeeze().permute(1, 2, 0).numpy()
            image2 = (image2 - image2.min()) / (image2.max() - image2.min())

            plt.subplot(2 * num_of_rows, 2 * num_of_columns, 2 * index + 1)
            plt.imshow(image1)
            plt.title("IMG-1")
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")

            plt.subplot(2 * num_of_rows, 2 * num_of_columns, 2 * index + 2)
            plt.imshow(image2)
            plt.title("IMG-2")
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["artifacts_path"], "images.jpeg"))
        print("Image stored in " + config()["path"]["artifacts_path"])
        plt.show()

    @staticmethod
    def dataset_details():
        train_dataloader = os.path.join(
            config()["path"]["processed_path"], "train_dataloader.pkl"
        )
        valid_dataloader = os.path.join(
            config()["path"]["processed_path"], "valid_dataloader.pkl"
        )

        train_dataloader = load(filename=train_dataloader)
        valid_dataloader = load(filename=valid_dataloader)

        total_train_dataset = sum(X1.size(0) for X1, _ in train_dataloader)
        total_valid_dataset = sum(X1.size(0) for X1, _ in valid_dataloader)

        total_dataset = total_train_dataset + total_valid_dataset

        train_dataset_dimension, _ = next(iter(train_dataloader))
        valid_dataset_dimension, _ = next(iter(valid_dataloader))

        dataset_details = pd.DataFrame(
            {
                "train_dataset_dimension": str(train_dataset_dimension.size()),
                "valid_dataset_dimension": str(valid_dataset_dimension.size()),
                "total_dataset": total_dataset,
                "train_dataset_size": total_train_dataset,
                "valid_dataset_size": total_valid_dataset,
                "total_dataset_size": total_dataset,
                "percentage_of_train_dataset": str(
                    total_train_dataset / total_dataset * 100
                    if total_train_dataset > 0
                    else 0
                )
                + "%",
                "percentage_of_valid_dataset": str(
                    total_valid_dataset / total_dataset * 100
                    if total_valid_dataset > 0
                    else 0
                )
                + "%",
            },
            index=["Dataset Details"],
        ).T

        dataset_details.to_csv(
            os.path.join(config()["path"]["artifacts_path"], "dataset_details.csv")
        )

        print(
            "Dataset details stored in the folder {}".capitalize().format(
                config()["path"]["artifacts_path"]
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataloader configuration for coupledGAN".title()
    )
    parser.add_argument(
        "--dataset",
        default=config()["dataloader"]["dataset"],
        help="Path to the dataset file".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        default=config()["dataloader"]["batch_size"],
        type=int,
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        default=config()["dataloader"]["image_size"],
        choices=[32, 64, 128],
        type=int,
        help="Th size of the image to be loaded".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        default=config()["dataloader"]["split_size"],
        type=int,
        help="The size of the split for the dataset".capitalize(),
    )

    args = parser.parse_args()

    loader = Loader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        split_size=args.split_size,
    )

    try:
        loader.unzip_folder()
    except ValueError as e:
        print(f"Error occurred in unzipping: {e}")
        exit(1)
    except Exception as e:
        print(f"Error occurred in unzipping: {e}")
        exit(1)

    try:
        loader.create_dataloader()
    except Exception as e:
        print(f"Error occurred creating data: {e}")
        exit(1)

    try:
        loader.display_images()
    except Exception as e:
        print(f"Error occurred display the image: {e}")
        exit(1)

    try:
        Loader.dataset_details()
    except Exception as e:
        print(f"Error occurred in detailing the dataset: {e}")
        exit(1)

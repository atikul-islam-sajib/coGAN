import os
import sys
import torch
import argparse

sys.path.append("./src/")

from utils import config
from trainer import Trainer
from dataloader import Loader
from generator import CoupledGenerators
from discriminator import CoupledDiscriminators


def cli():
    parser = argparse.ArgumentParser(
        description="CLI configuration for coupledGAN".title()
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
    parser.add_argument(
        "--epochs",
        default=config()["trainer"]["epochs"],
        type=int,
        help="Number of epochs for training".capitalize(),
    )
    parser.add_argument(
        "--lr",
        default=config()["trainer"]["lr"],
        type=float,
        help="Defines the learning rate".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        default=config()["trainer"]["momentum"],
        type=float,
        help="Defines the momentum for SGD".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        default=config()["trainer"]["beta1"],
        type=float,
        help="Defines the beta1 for Adam".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        default=config()["trainer"]["beta1"],
        type=float,
        help="Defines the beta2 for Adam".capitalize(),
    )
    parser.add_argument(
        "--regularizer",
        default=config()["trainer"]["regularizer"],
        type=float,
        help="Regularization term for the loss function".capitalize(),
    )
    parser.add_argument(
        "--device",
        default=config()["trainer"]["device"],
        choices=["cpu", "cuda"],
        help="Device for training".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Use Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="Use SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["trainer"]["l1_regularization"],
        help="Use L1 regularization".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["trainer"]["l2_regularization"],
        help="Use L2 regularization".capitalize(),
    )
    parser.add_argument(
        "--elasticnet_regularization",
        type=bool,
        default=config()["trainer"]["elasticnet_regularization"],
        help="Use ElasticNet regularization".capitalize(),
    )
    parser.add_argument(
        "--mlflow",
        type=bool,
        default=config()["trainer"]["mlflow"],
        help="Enable logging to MLflow".capitalize(),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config()["trainer"]["verbose"],
        help="Display progress during training".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    loader = Loader(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    try:
        loader.unzip_folder()
    except ValueError as e:
        print(f"Error occurred: {e}")
        exit(1)
    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)

    try:
        loader.create_dataloader()
    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)

    try:
        loader.display_images()
    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)

    try:
        Loader.dataset_details()
    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)

    if args.train:
        trainer = Trainer(
            epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            beta1=args.beta1,
            beta2=args.beta2,
            regularizer=args.regularizer,
            device=args.device,
            adam=args.adam,
            SGD=args.SGD,
            l1_regularization=args.l1_regularization,
            l2_regularization=args.l2_regularization,
            elasticnet_regularization=args.elasticnet_regularization,
            mlflow=args.mlflow,
            verbose=args.verbose,
        )

        trainer.train()
        Trainer.model_history()


if __name__ == "__main__":
    cli()

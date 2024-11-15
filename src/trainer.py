import os
import sys
import torch
import mlflow
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

sys.path.append("./src/")

from utils import config, load, device_init, dump
from helper import helper
from loss import AdversarialLoss
from generator import CoupledGenerators
from discriminator import CoupledDiscriminators


class Trainer:
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 2e-4,
        momentum: float = 0.75,
        beta1: float = 0.5,
        beta2: float = 0.999,
        regularizer: float = 1e-2,
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
        self.regularizer = regularizer
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticnet_regularization = elasticnet_regularization
        self.mlflow = mlflow
        self.verbose = verbose

        self.init = helper(
            adam=self.adam,
            SGD=self.SGD,
            beta1=self.beta1,
            beta2=self.beta2,
            lr=self.lr,
            momentum=self.momentum,
            reduction="mean",
        )

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.netG = self.init["netG"]
        self.netD = self.init["netD"]

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD = self.init["optimizerD"]

        self.adversarial_loss = self.init["adversarial_loss"]

        assert (
            self.train_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Train Dataloader should be the class of dataloader".capitalize()
        assert (
            self.valid_dataloader.__class__ == torch.utils.data.DataLoader
        ), "Validation Dataloader should be the class of dataloader".capitalize()

        self.netG.__class__ == CoupledGenerators, "Generator should be the class of CoupledGenerators".capitalize()
        self.netD.__class__ == CoupledDiscriminators, "Discriminator should be the class of CoupledDiscriminators".capitalize()

        if self.adam:
            assert (
                self.optimizerG.__class__ == optim.Adam
            ), "OptimizerG should be the class of Adam".capitalize()
            assert (
                self.optimizerD.__class__ == optim.Adam
            ), "OptimizerD should be the class of Adam".capitalize()
        else:
            assert (
                self.optimizerG.__class__ == optim.SGD
            ), "OptimizerG should be the class of SGD".capitalize()
            assert (
                self.optimizerD.__class__ == optim.SGD
            ), "OptimizerD should be the class of SGD".capitalize()

        self.adversarial_loss.__class__ == AdversarialLoss, "AdversarialLoss should be the class of Adversarial".capitalize()

        self.device = device_init(device=self.device)

        self.loss = float("inf")
        self.history = {"netG_loss": [], "netD_loss": []}

    def l1_regularizer(self, model):
        if model is not None:
            return self.regularizer * sum(
                torch.norm(params, 1) for params in model.parameters()
            )

    def l2_regularizer(self, model):
        if model is not None:
            return self.regularizer * sum(
                torch.norm(params, 2) ** 2 for params in model.parameters()
            )

    def elasticnet_regularizer(self, model):
        if model is not None:
            return self.regularizer * sum(
                torch.norm(params, 1) + 0.5 * torch.norm(params, 2) ** 2
                for params in model.parameters()
            )

    def saved_checkpoints(self, **kwargs):
        os.makedirs(config()["path"]["train_models"], exist_ok=True)
        os.makedirs(config()["path"]["test_model"], exist_ok=True)

        train_loss = kwargs["train_loss"]
        valid_loss = kwargs["valid_loss"]

        epoch = kwargs["epoch"]

        if self.loss > train_loss:
            self.loss = train_loss
            torch.save(
                {
                    "netG": self.netG.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "epoch": epoch + 1,
                },
                os.path.join(
                    config()["path"]["test_model"],
                    "best.pth",
                ),
            )

        torch.save(
            self.netG.state_dict(),
            os.path.join(
                config()["path"]["train_models"],
                "netG{}.pth".format(epoch + 1),
            ),
        )

    def update_netG_training(self, **kwargs):
        self.optimizerG.zero_grad()

        latent_dim = kwargs["latent_dim"]

        generated_image1, generated_image2 = self.netG(latent_dim)

        predicted_image1, predicted_image2 = self.netD(
            generated_image1, generated_image2
        )

        predicted_image1_loss = self.adversarial_loss(
            predicted_image1, torch.ones_like(predicted_image1)
        )
        predicted_image2_loss = self.adversarial_loss(
            predicted_image2, torch.ones_like(predicted_image2)
        )

        total_loss = (predicted_image1_loss + predicted_image2_loss) / 2

        if self.l1_regularization:
            total_loss += self.l1_regularizer(model=self.netG)
        elif self.l2_regularization:
            total_loss += self.l2_regularizer(model=self.netG)
        elif self.elasticnet_regularization:
            total_loss += self.elasticnet_regularizer(model=self.netG)

        total_loss.backward()
        self.optimizerG.step()

        return total_loss.item()

    def update_netD_training(self, **kwargs):
        self.optimizerD.zero_grad()

        image1 = kwargs["image1"]
        image2 = kwargs["image2"]

        latent_dim = kwargs["latent_dim"]

        generated_image1, generated_image2 = self.netG(latent_dim)

        predicted_generated_image1, predicted_generated_image2 = self.netD(
            generated_image1, generated_image2
        )

        predicted_actual_image1, predicted_actual_image2 = self.netD(image1, image2)

        predicted_generated_image1_loss = self.adversarial_loss(
            predicted_generated_image1, torch.zeros_like(predicted_generated_image1)
        )
        predicted_generated_image2_loss = self.adversarial_loss(
            predicted_generated_image2, torch.zeros_like(predicted_generated_image2)
        )

        predicted_actual_image1_loss = self.adversarial_loss(
            predicted_actual_image1, torch.ones_like(predicted_actual_image1)
        )
        predicted_actual_image2_loss = self.adversarial_loss(
            predicted_actual_image2, torch.ones_like(predicted_actual_image2)
        )

        total_loss = (
            predicted_generated_image1_loss
            + predicted_generated_image2_loss
            + predicted_actual_image1_loss
            + predicted_actual_image2_loss
        ) / 4

        total_loss.backward()
        self.optimizerD.step()

        return total_loss.item()

    def display_progress(self, **kwargs):
        epoch = kwargs["epoch"]
        train_loss = kwargs["train_loss"]
        valid_loss = kwargs["valid_loss"]

        if self.verbose:
            print(
                "Epochs: [{}/{}] - train_loss: {:.4f} - test_loss: {:.4f}".format(
                    epoch + 1, self.epochs, train_loss, valid_loss
                )
            )
        else:
            print(f"Epoch: {epoch + 1}/{self.epochs} is completed".capitalize())

    def train(self):
        experiment_name = "coupledGAN - version 0.0.12"
        experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            for _, epoch in tqdm(enumerate(range(self.epochs))):
                netG_loss = []
                netD_loss = []
                for idx, (image1, image2) in enumerate(self.train_dataloader):
                    image1 = image1.to(self.device)
                    image2 = image2.to(self.device)

                    batch_size = image1.size(0)
                    latent_dim = torch.randn(
                        (batch_size, config()["netG"]["latent_space"])
                    ).to(self.device)

                    netG_loss.append(
                        self.update_netG_training(
                            image1=image1, image2=image2, latent_dim=latent_dim
                        )
                    )
                    netD_loss.append(
                        self.update_netD_training(
                            image1=image1, image2=image2, latent_dim=latent_dim
                        )
                    )

                try:
                    self.display_progress(
                        epoch=epoch,
                        train_loss=np.mean(netG_loss),
                        valid_loss=np.mean(netD_loss),
                    )
                except Exception as e:
                    print(
                        f"Error occurred during training in display progress: {str(e)}"
                    )
                    exit(1)

                try:
                    batch_size = config()["dataloader"]["batch_size"]
                    latent_dim = config()["netG"]["latent_space"]

                    Z = torch.randn((batch_size, latent_dim))

                    image1, image2 = self.netG(Z)

                    train_results = config()["path"]["train_results"]

                    for filename, image in [("image1", image1), ("image2", image2)]:
                        save_image(
                            image,
                            os.path.join(train_results, f"{filename}{epoch + 1}.png"),
                        )

                except Exception as e:
                    print(f"Error occurred during training in saving images: {str(e)}")
                    exit(1)

                try:
                    self.saved_checkpoints(
                        train_loss=np.mean(netG_loss),
                        valid_loss=np.mean(netD_loss),
                        epoch=epoch,
                    )
                except Exception as e:
                    print(
                        f"Error occurred during training in saved checkpoints: {str(e)}"
                    )
                    exit(1)

                try:
                    self.history["netG_loss"].append(np.mean(netG_loss))
                    self.history["netD_loss"].append(np.mean(netD_loss))
                except Exception as e:
                    print(f"Error occurred during training in model_history: {str(e)}")
                    exit(1)

                try:
                    mlflow.log_params(
                        {
                            "epochs": self.epochs,
                            "lr": self.lr,
                            "momentum": self.momentum,
                            "beta1": self.beta1,
                            "beta2": self.beta2,
                            "regularizer": self.regularizer,
                            "adam": self.adam,
                            "SGD": self.SGD,
                            "device": self.device,
                            "l1_regularization": self.l1_regularization,
                            "l2_regularization": self.l2_regularization,
                            "elasticnet_regularization": self.elasticnet_regularization,
                            "verbose": self.verbose,
                        }
                    )
                except Exception as e:
                    print(f"An error occurred while logging parameters to MLflow: {e}")

                try:
                    mlflow.log_metric(
                        key="netG_loss", value=np.mean(netG_loss), step=epoch + 1
                    )
                    mlflow.log_metric(
                        key="netD_loss", value=np.mean(netD_loss), step=epoch + 1
                    )
                except Exception as e:
                    print(f"An error occurred while logging metrics to MLflow: {e}")

            try:
                dump(
                    value=self.history,
                    filename=os.path.join(
                        config()["path"]["metrics_path"], "history.pkl"
                    ),
                )
            except Exception as e:
                print(
                    f"Error occurred during training in training_completion: {str(e)}"
                )
                exit(1)

            try:
                mlflow.pytorch.log_model(self.netG, "coupledGAN")
            except Exception as e:
                print(
                    f"Error occurred during the saving the model using MLFlow: {str(e)}"
                )
                exit(1)

    @staticmethod
    def model_history():
        metrics = os.path.join(config()["path"]["metrics_path"], "history.pkl")
        metrics = load(filename=metrics)

        _, axes = plt.subplots(1, 2, figsize=(15, 10))

        axes[0].plot(metrics["netG_loss"], label="Generator Loss")
        axes[0].legend()
        axes[0].set_title("Generator Loss")
        axes[0].set_xlabel("Epochs")
        axes[1].set_ylabel("Generator Loss")

        axes[1].plot(metrics["netD_loss"], label="Discriminator Loss")
        axes[1].legend()
        axes[1].set_title("Discriminator Loss")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Discriminator Loss")

        plt.tight_layout()
        plt.savefig(
            os.path.join(config()["path"]["metrics_path"], "learning_curves.jpeg")
        )
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for the coupledGAN".title())
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

    args = parser.parse_args()

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

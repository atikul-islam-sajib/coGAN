import os
import sys
import math
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("./src/")

from utils import config, load, device_init
from generator import CoupledGenerators


class Tester:
    def __init__(self, quantity: int = 64, model: str = "best", device: str = "cuda"):
        self.quantity = quantity
        self.model = model
        self.device = device

        self.device = device_init(device = self.device)

    def select_the_best_model(self):
        if (os.path.exists(config()["path"]["test_model"])) and (
            os.path.exists(config()["path"]["train_models"])
        ):
            best_model_path = os.path.join(config()["path"]["test_model"], "best.pth")
            selected_model_path = os.path.join(
                config()["path"]["train_models"], self.model
            )

            print(best_model_path)

            if self.model == "best":
                best_model = torch.load(best_model_path)
                best_model = best_model["netG"]
                return best_model
            else:
                return torch.load(selected_model_path)

        else:
            raise FileNotFoundError(
                "The test_model or train_models directory does not exist.".capitalize()
            )

    def load_dataset(self):
        if self.dataset == "valid":
            return load(
                filename=os.path.join(
                    config()["path"]["processed_path"], "valid_dataloader.pkl"
                )
            )
        else:
            return load(
                filename=os.path.join(
                    config()["path"]["processed_path"], "valid_dataloader.pkl"
                )
            )

    def test(self):
        try:
            self.netG = CoupledGenerators(
                latent_space=config()["netG"]["latent_space"],
                constant=config()["netG"]["constant"],
                image_size=config()["dataloader"]["image_size"],
            )

            self.netG.load_state_dict(self.select_the_best_model())

            self.netG.to(self.device)

            plt.figure(figsize=(20, 15))

            try:
                Z = torch.randn((self.quantity, config()["netG"]["latent_space"]))

                generated_image1, generated_image2 = self.netG(Z.to(self.device))

                num_of_rows = int(math.sqrt(generated_image1.size(0)))
                num_of_cols = generated_image1.size(0) // num_of_rows

                for index, image in enumerate(generated_image1):
                    image1 = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                    image1 = (image1 - image1.min()) / (image1.max() - image1.min())

                    image2 = (
                        generated_image2[index]
                        .squeeze()
                        .permute(1, 2, 0)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    image2 = (image2 - image2.min()) / (image2.max() - image2.min())

                    plt.subplot(2 * num_of_rows, 2 * num_of_cols, 2 * index + 1)
                    plt.imshow(image1)
                    plt.title("G:IMG-1")
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis("off")

                    plt.subplot(2 * num_of_rows, 2 * num_of_cols, 2 * index + 2)
                    plt.imshow(image2)
                    plt.title("G:IMG-2")
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis("off")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(config()["path"]["test_result"], "images.jpeg")
                )
                print(
                    "Test image saved in the folder {}".format(
                        config()["path"]["test_result"]
                    )
                )
                plt.show()

            except Exception as e:
                print(f"Error occurred in the generating image: {e}")

        except FileExistsError as e:
            print(f"Error occurred in the file: {e}")
            return None
        except Exception as e:
            print(f"Error occurred in the file: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester code for coupledGAN".title())
    parser.add_argument(
        "--quantity",
        type=int,
        default=config()["tester"]["quantity"],
        help="Define the number of synthetic data that you want to create".capitalize(),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config()["tester"]["model"],
        help="Define which model you want to".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Define the device to use".capitalize(),
    )

    args = parser.parse_args()

    tester = Tester(quantity=args.quantity, model=args.model, device=args.device)

    tester.test()

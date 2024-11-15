# Fake Image Generator - coupledGAN (Coupled Generative Adversarial Networks)

![CoupledGAN](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*JY9G6jB3CRpjQ2fN3yBugA.png)

Coupled Generative Adversarial Networks (CoupledGAN or CoGAN) is a framework designed to learn a joint representation of multiple domains without requiring paired examples during training. It is particularly useful in scenarios where the data from two different domains share some common underlying structure but are not aligned (e.g., images of the same objects in different styles, languages, or modalities).

![CoupledGAN](https://github.com/atikul-islam-sajib/Research-Assistant-Work-HWR/blob/main/fake_image.png)

## Key Features of CoupledGAN (CoGAN):

1. **Dual Generators and Discriminators:**
   - CoGAN consists of two (or more) GANs, each responsible for modeling data in its own domain.
   - Each GAN has its own generator \(G1\), \(G2\) and discriminator \(D1\), \(D2\).

2. **Weight Sharing:**
   - The networks enforce the sharing of certain layers in the generators and discriminators (typically the early layers in the generator and the later layers in the discriminator). This allows CoGAN to capture a shared latent representation across domains.

## Features

- Utilizes PyTorch for implementing GAN models.
- Provides scripts for easy training and generating synthetic images.
- Command Line Interface for easy interaction.
- Includes a custom data loader for the Custom dataset.
- Customizable training parameters for experimenting with GAN.

## Installation

Clone the repository:

```
git clone https://github.com/atikul-islam-sajib/coGAN.git

cd coGAN
```

# Install dependencies

```
pip install -r requirements.txt
```

## Usage

Examples of commands and their explanations.

```bash
python /path/to/coGAN/src/cli.py --help
```

### Options

- `--batch_size BATCH_SIZE`: Set the batch size for the dataloader. (Default: specify if there's one)
- `--image_path`: Define the dataset path.
- `--epochs EPOCHS`: Set the number of training epochs.
- `--latent_space LATENT_SPACE`: Define the size of the latent space for the model.
- `--lr LR`: Specify the learning rate for training the model.
- `--quantity SAMPLES`: Determine the number of samples to generate after training.
- `--test`: Run tests with synthetic data to validate model performance.
- `--device`: Train the model with CPU, GPU, MPS.
- `--verbose`: Display the critic loss and generator loss in each iterations[True/False]

## Training and Generating Images(CLI)

### Training the GAN Model

To train the GAN model with default parameters with mps:

```
!python /content/coGAN/src/cli.py  --epochs 200 --latent_space 100 --image_size 64  --lr 0.00005 --device cuda --batch_size 64 --dataset /image.zip/ --display True --train
```

To train the GAN model with default parameters with cpu:

```
!python /content/coGAN/src/cli.py  --epochs 200 --latent_space 100 --image_size 64  --lr 0.00005 --device cuda --batch_size 64 --dataset /image.zip/ --verbose True --train
```


### Generating Images

To generate images using the trained model:

```
!python /content/coGAN/src/cli.py --quantity 20 --device cuda --test
```

### Viewing Generated Images

Check the specified output directory for the generated images.

```
from IPython.display import Image
Image(filename='/content/coGAN/artifacts/outputs/train_results/XYZ.png')
```

## Core Script Usage

The core script sets up the necessary components for training the GAN. Here's a quick overview of what each part does:

```python
from src.dataloader import Loader
from src.generator import CoupledGenerators
from src.discriminator import CoupledDiscriminators
from src.trainer import Trainer
from src.tester import Tester

# Initialize the data loader with batch size
loader = Loader(
    dataset = "/content/drive/MyDrive/anime.zip",
    batch_size = 128,
    image_size = 64,
    split_size = 0.25
)

    loader.unzip_folder()
    loader.create_dataloader()

#================================================================================================================#

# Set up the trainer with learning rate, epochs, and latent space size
trainer = Trainer(
    lr = 0.0002,
    epochs = 20,
    verbose = True

    ... ... ... 
    ... ... ...
    ... ... ...
)

trainer.train()

#================================================================================================================#

# Test the generated dataset and display the synthetic images
tester = Tester(
    model="best",
    quantity = 64,
    device="cuda"

    ... ... ...
    ... ... ...
)
tester.test()

#================================================================================================================#

from IPython.display import Image
Image("/content/coGAN/outputs/artifacts/test_result/XYZ.png")
```

This script initializes the data loader, downloads the Custom dataset, and prepares the data loader. It then sets up and starts the training process for the GAN model.

## Notebook Training

For detailed documentation on the implementation and usage using notebook, visit the [Notebook for CLI](https://github.com/atikul-islam-sajib/coGAN/blob/main/notebooks/coGAN_Tutorial.ipynb).

For detailed documentation on the implementation and usage using notebook, visit the [Notebook for ModelPrototype](./notebooks/ModelPrototype.ipynb).

## Contributing

Contributions to improve the project are welcome. Please follow the standard procedures for contributing to open-source projects.

## License

This project is licensed under [MIT LICENSE](./LICENSE). Please see the LICENSE file for more details.

## Contact

For any inquiries or suggestions, feel free to reach out to [atikulislamsajib137@gmail.com].

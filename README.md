# im2im-uq
A platform for image-to-image translation with rigorous, distribution-free uncertainty quantification.

## Installation
You will need to execute
```
conda env create -f environment.yml
conda activate microcv
```
You will also need to go through the Weights and Biases setup process, which involves defining the following environment variables:
```
WANDB_API_KEY
WANDB_USERNAME
WANDB_USER_EMAIL
```
## Running an experiment
```
wandb sweep experiments/test/configs-default.yaml
```
Then run the wandb agent given.

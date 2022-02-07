# im2im-uq
A platform for image-to-image regression with rigorous, distribution-free uncertainty quantification.

## Summary
This repository provides a convenient way to train deep-learning models in PyTorch for image-to-image regression---any task where the input and output are both images---along with rigorous uncertainty quantification.
The uncertainty quantification takes the form of an interval for each pixel which is guaranteed to contain most true pixel values with high-probability no matter the choice of model or the dataset used (it is a _risk-controlling prediction set_). 
The training pipeline is already built to handle more than one GPU and all training/calibration should run automatically.

The basic idea is

* Define your dataset in ```core/datasets/```.
* Create a folder for your experiment ```experiments/new_experiment```, along with a file ```experiments/new_experiment/config.yml``` defining the model architecture, hyperparameters, and method of uncertainty quantification. You can use ```experiments/fastmri/config.yml``` as a template.
* From the root folder, run ```wandb sweep experiments/new_experiment/config.yml```, and run the resulting sweep.

Following this procedure will train one or more models (depending on ```config.yml```) that perform image-to-image regression with rigorous uncertainty quantification.

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

## Adding new models and datasets
* Define the "trunk" of your model (everything but the last layer) in ```core/models/trunks```
* Define the final layer of your model in ```core/models/finallayers```.  The final layer must output a lower-endpoint, prediction, and upper-endpoint for each pixel, defining an uncertainty interval for use in ```core/models/add_uncertainty.py```.
* Define an indexed sequence of nested sets 

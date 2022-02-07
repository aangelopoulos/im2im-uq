# im2im-uq
A platform for image-to-image regression with rigorous, distribution-free uncertainty quantification.

## Summary
This repository provides a convenient way to train deep-learning models in PyTorch for image-to-image regression---any task where the input and output are both images---along with rigorous uncertainty quantification.
The uncertainty quantification takes the form of an interval for each pixel which is guaranteed to contain most true pixel values with high-probability no matter the choice of model or the dataset used (it is a _risk-controlling prediction set_). 
The training pipeline is already built to handle more than one GPU and all training/calibration should run automatically.

The basic idea is

* Define your dataset in ```core/datasets/```.
* Create a folder for your experiment ```experiments/new_experiment```, along with a file ```experiments/new_experiment/config.yml``` defining the model architecture, hyperparameters, and method of uncertainty quantification. You can use ```experiments/fastmri_test/config.yml``` as a template.
* From the root folder, run ```wandb sweep experiments/new_experiment/config.yml```, and run the resulting sweep.
* After the sweep is complete, models will be saved in ```experiments/new_experiment/checkpoints```, the metrics will be printed to the terminal, and outputs will be in ```experiments/new_experiment/output/```.  See ```experiments/fastmri_test/plot.py``` for an example of how to make plots from the raw outputs.

Following this procedure will train one or more models (depending on ```config.yml```) that perform image-to-image regression with rigorous uncertainty quantification.

There are two pre-baked examples that you can run on your own after downloading the open-source data: ```experiments/fastmri_test/config.yml``` and ```experiments/temca_test/config.yml```.
The third pre-baked example, ```experiments/bsbcm_test/config.yml```, reiles on data collected at Berkeley that has not yet been publicly released (but will be soon).

## Paper 
[Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging](https://arxiv.org/abs/????.?????)
```
@article{angelopoulos2022image,
  title={Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging},
  author={Angelopoulos, Anastasios N and Kohli, Amit P and Bates, Stephen and Jordan, Michael I and Malik, Jitendra and Alshaabi, Thayer and Upadhyayula, Srigokul and Romano, Yaniv},
  journal={arXiv preprint arXiv:????.?????},
  year={2022}
}
```

## Installation
You will need to execute
```
conda env create -f environment.yml
conda activate microcv
```
You will also need to go through the Weights and Biases setup process that initiates when you run your first sweep.

## Reproducing the results

## Adding a new experiment
If you want to extend this code to 

## Adding new datasets
To add a new dataset, use the following procedure.
* 

* Define the "trunk" of your model (everything but the last layer) in ```core/models/trunks```
* Define the final layer of your model in ```core/models/finallayers```.  The final layer must output a lower-endpoint, prediction, and upper-endpoint for each pixel, defining an uncertainty interval for use in ```core/models/add_uncertainty.py```.
* Define an indexed sequence of nested sets 

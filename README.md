# microcv
A library for computer vision and microscopy.

Todo List:
- [x] Make dir structure
- [x] Make base environment YML (numpy, torch/torchvision/cuda/etc, pandas, opencv, matplotlib, seaborn, celluloid, tensorboard/weights&biases, jupyter, skimage, scipy, pickle, h5py, imageio, tqdm, pip.
- [ ] setup main.py with W&B as script router, will have a corresponding main config file for experiment-wide paramters (e.g., file paths). This specifies which experiment to run.
- [ ] Set up a experiment run script. These scripts, called by main, will grab a dataset and model trunk, split the dataset, modify the model and call train.py. Each has its own config file to specify which trunk and dataset as well as which uncertainty method and any relevant parameters.
- [ ] Set up a model trunk (e.g., simple U-Net) which will instantiated by an experiment run script
- [ ] Set up a dataset file (CARE_dataset.py) which will return a pytorch dataset and take in number of instances and any dataset specific params
- [ ] Set up train script which recieves a model, train and val dataloaders, and training parameters and logs using W&B.
- [ ] Write test.py which evaluates the performance of a model on a test dataloader. This also will get called by an experiment run script. 
- [ ] Train example model on a few instances of a dataset to test full pipeline

- [ ] Now write add_uncertainty.py which takes in a model trunk and string and modified the model for a particular uncertainty method (start with simple ones)
- [ ] Write calibrate.py which takes in model, calibration dataloader and returns a score threshold which allows for the model to return good sets.
- [ ] modify test.py to add a few metrics for checking coverage, etc
- [ ] Now run the full pipeline: make a experiment run script which grabs a model trunk and dataset, adds an uncertainty method to the model via add_uncertainty.py, calls train.py to train it, calls calibrate.py will calibration data to calibrate the model, and finally runs test.py to test the models performance and uncertainty.
- [ ] Add a visualize.py which has a bunch of methods for various plots and visualizations given a trained model and some data (uncertainty visualization, coverage statistics, etc).

## Installation
You will need to execute
```
conda env create -f environment.yml
conda activate microcv
```
You will also need to define the following environment variables:
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

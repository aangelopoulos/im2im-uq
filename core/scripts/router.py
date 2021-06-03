import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import wandb
import numpy as np
import torch
import torchvision 
import torchvision.transforms as T
import yaml

from core.scripts.train import train_net
from core.models.add_uncertainty import add_uncertainty

# Models
from core.models.trunks.unet import UNet

# Datasets
from core.datasets.CAREDrosophila import CAREDrosophilaDataset

if __name__ == "__main__":
  wandb.init() 

  # DATASET LOADING
  if wandb.config["dataset"] == "CIFAR10":
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = T.Compose([ T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize ])
    dataset = torchvision.datasets.CIFAR10('/clusterfs/abc/angelopoulos/CIFAR10', download=True, transform=transform)
  if wandb.config["dataset"] == "CAREDrosophila":
    path = '/clusterfs/abc/angelopoulos/care/Isotropic_Drosophila/train_data/data_label.npz'
    dataset = CAREDrosophilaDataset(path, num_instances=500, normalize='min-max')
  else:
    raise NotImplementedError 

  # MODEL LOADING
  if wandb.config["dataset"] == "CIFAR10":
    if wandb.config["model"] == "ResNet18":
      trunk = torchvision.models.resnet18(num_classes=wandb.config["num_classes"])
  if wandb.config["model"] == "UNet":
      trunk = UNet(1,1)

  # ADD LAST LAYER OF MODEL
  model = add_uncertainty(trunk, wandb.config)

  # DATA SPLITTING
  lengths = np.round(len(dataset)*np.array(wandb.config["data_split_percentages"])).astype(int)
  lengths[-1] = len(dataset)-(lengths.sum()-lengths[-1])
  train_dataset, calib_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths.tolist()) 
  
  train_net(model,
            train_dataset,
            val_dataset,
            device=wandb.config["device"],
            epochs=wandb.config["epochs"],
            batch_size=wandb.config["batch_size"],
            lr=wandb.config["lr"],
            checkpoint_dir=wandb.config["checkpoint_dir"],
            checkpoint_every=wandb.config["checkpoint_every"],
            validate_every=wandb.config["validate_every"])  

  curr_lr = wandb.config["lr"]
  print(f"Done with lr={curr_lr}")

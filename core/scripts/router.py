import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import wandb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as T
import yaml

from core.scripts.train import train_net
from core.scripts.eval import eval_net, eval_risk_size 
from core.models.add_uncertainty import add_uncertainty
from core.calibration.calibrate_model import calibrate_model

# Models
from core.models.trunks.unet import UNet

# Datasets
from core.datasets.CAREDrosophila import CAREDrosophilaDataset

if __name__ == "__main__":
  wandb.init() 
  curr_lr = wandb.config["lr"]

  # DATASET LOADING
  if wandb.config["dataset"] == "CIFAR10":
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = T.Compose([ T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize ])
    dataset = torchvision.datasets.CIFAR10('/clusterfs/abc/angelopoulos/CIFAR10', download=True, transform=transform)
  if wandb.config["dataset"] == "CAREDrosophila":
    path = '/clusterfs/abc/angelopoulos/care/Isotropic_Drosophila/train_data/data_label.npz'
    dataset = CAREDrosophilaDataset(path, num_instances='all', normalize='min-max')
  else:
    raise NotImplementedError 

  # MODEL LOADING
  if wandb.config["dataset"] == "CIFAR10":
    if wandb.config["model"] == "ResNet18":
      trunk = torchvision.models.resnet18(num_classes=wandb.config["num_classes"])
  if wandb.config["model"] == "UNet":
      trunk = UNet(1,1)

  # ADD LAST LAYER OF MODEL
  params = { key: wandb.config[key] for key in wandb.config.keys() }
  print(params)
  model = add_uncertainty(trunk, params)

  # DATA SPLITTING
  lengths = np.round(len(dataset)*np.array(wandb.config["data_split_percentages"])).astype(int)
  lengths[-1] = len(dataset)-(lengths.sum()-lengths[-1])
  train_dataset, calib_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths.tolist()) 
  
  model = train_net(model,
                    train_dataset,
                    val_dataset,
                    wandb.config['device'],
                    wandb.config['epochs'],
                    wandb.config['batch_size'],
                    wandb.config['lr'],
                    wandb.config['load_from_checkpoint'],
                    wandb.config['checkpoint_dir'],
                    wandb.config['checkpoint_every'],
                    wandb.config['validate_every'],
                    params)   

  print("Done training!")
  model.eval()
  val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
  val_loss = eval_net(model,val_loader,wandb.config['device'])
  print(f"Done validating! Validation Loss: {val_loss}")
  model = calibrate_model(model, calib_dataset, params)
  print(f"Model calibrated! lambda hat = {model.lhat}")
  risk, sizes = eval_risk_size(model, val_dataset, params)
  print(f"Risk: {risk}  |  Mean size: {sizes.mean()}")

  print(f"Done with lr={curr_lr}")

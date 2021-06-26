import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
import torch.nn as nn
from core.datasets.CAREDrosophila import CAREDrosophilaDataset
from core.datasets.fastmri import FastMRIDataset
from core.models.trunks.unet import UNet
from core.models.trunks.wnet import WNet
from core.models.add_uncertainty import add_uncertainty
from core.calibration.calibrate_model import calibrate_model 
from core.utils import fix_randomness 
from core.datasets.utils import normalize_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from core.scripts.train import train_net
from core.scripts.eval import get_images, eval_net, eval_set_metrics
import yaml
import pdb

import wandb

if __name__ == "__main__":
    fix_randomness()
    wandb.init()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/config.yml') as file:
      config = yaml.safe_load(file)
    if config["dataset"] == "CAREDrosophila":
      path = '/clusterfs/abc/angelopoulos/care/Isotropic_Drosophila/train_data/data_label.npz'
      dataset = CAREDrosophilaDataset(path, num_instances='all', normalize='min-max')
      #dataset = normalize_dataset(dataset)
    elif config["dataset"] == "fastmri":
      path = '/clusterfs/abc/amit/fastmri/knee/singlecoil_train/'
      mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
      dataset = FastMRIDataset(path, normalize_input='standard', normalize_output = 'min-max', mask_info=mask_info, num_volumes=300)
      dataset = normalize_dataset(dataset)
      config.update(dataset.norm_params)
    trunk = UNet(1,1)
    model = add_uncertainty(trunk, config)
    lengths = np.round(len(dataset)*np.array(config["data_split_percentages"])).astype(int)
    lengths[-1] = len(dataset)-(lengths.sum()-lengths[-1])
    train_dataset, calib_dataset, val_dataset, _ = random_split(dataset, lengths.tolist())

    model = train_net(model,
                      train_dataset,
                      val_dataset,
                      config['device'],
                      config['epochs'],
                      config['batch_size'],
                      config['lr'],
                      config['load_from_checkpoint'],
                      config['checkpoint_dir'],
                      config['checkpoint_every'],
                      config['validate_every'],
                      config)   
    model.eval()
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    #val_loss = eval_net(model,val_loader,config['device'])
    #print(f"Done validating! Validation Loss: {val_loss}")
    model = calibrate_model(model, calib_dataset, config)
    print(f"Model calibrated! lambda hat = {model.lhat}")
    # Get the prediction sets and properly organize them 
    examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth = get_images(model,
                                                                                                                      val_dataset,
                                                                                                                      config['device'],
                                                                                                                      list(range(5)))
    # Log everything
    wandb.log({"epoch": config['epochs']+1, "examples_input": examples_input})
    wandb.log({"epoch": config['epochs']+1, "Lower edge": examples_lower_edge})
    wandb.log({"epoch": config['epochs']+1, "Predictions": examples_prediction})
    wandb.log({"epoch": config['epochs']+1, "Upper edge": examples_upper_edge})
    wandb.log({"epoch": config['epochs']+1, "Ground truth": examples_ground_truth})
    # Evaluate the risk and size
    print(f"input shape: {val_dataset[0][0].shape}")
    risk, sizes, spearman, stratified_risk = eval_set_metrics(model, val_dataset, config)
    print(f"Risk: {risk}  |  Mean size: {sizes.mean()}  |  Spearman: {spearman}  |  stratified risk: {stratified_risk}  ")
    wandb.log({"risk": risk, "mean_size":sizes.mean(), "Spearman":spearman, "Size-Stratified Risk":stratified_risk})

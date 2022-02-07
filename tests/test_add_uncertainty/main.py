import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import random, copy
import torch
import torch.nn as nn
from core.datasets.fastmri import FastMRIDataset
from core.datasets.bsbcm import BSBCMDataset
from core.datasets.temca import TEMCADataset
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
    if config["dataset"] == "bsbcm":
      path = '/home/aa/data/bsbcm'
      dataset = BSBCMDataset(path, num_instances='all', normalize=config["output_normalization"])
      num_inputs = 2
    elif config["dataset"] == "fastmri":
      path = '/clusterfs/abc/angelopoulos/fastmri/knee/singlecoil_train/'
      mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
      dataset = FastMRIDataset(path, normalize_input=config["input_normalization"], normalize_output = config["output_normalization"], mask_info=mask_info)
      dataset = normalize_dataset(dataset)
      config.update(dataset.norm_params)
      num_inputs = 1 
    elif config["dataset"] == "temca":
      path = '/clusterfs/fiona/amit/temca_data/'
      dataset = TEMCADataset(path, patch_size=[320,320], downsampling=[4,4], num_imgs='all', buffer_size=5, normalize='01') 
      num_inputs = 1 
    else:
      raise NotImplementedError

    trunk = UNet(num_inputs,1)
    model = add_uncertainty(trunk, config)
    if config["dataset"] == "temca":
      img_paths = dataset.img_paths
      lengths = np.round(len(img_paths)*np.array(config["data_split_percentages"])).astype(int)
      lengths[-1] = len(img_paths)-(lengths.sum()-lengths[-1])
      random.shuffle(img_paths)
      train_dataset = copy.deepcopy(dataset)
      calib_dataset = copy.deepcopy(dataset)
      val_dataset = copy.deepcopy(dataset)
      train_dataset.img_paths = img_paths[:lengths[0]]
      calib_dataset.img_paths = img_paths[lengths[0]:(lengths[0]+lengths[1])]
      val_dataset.img_paths = img_paths[(lengths[0]+lengths[1]):(lengths[0]+lengths[1]+lengths[2])]
    else:
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
    model, _ = calibrate_model(model, calib_dataset, config)
    print(f"Model calibrated! lambda hat = {model.lhat}")
    # Get the prediction sets and properly organize them 
    """
    examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth, examples_lower_length, examples_upper_length, _ = get_images(model,
                                                                                                                                                                        val_dataset,
                                                                                                                                                                       config['device'],
                                                                                                                                                                       list(range(5)),
                                                                                                                                                                       config)
    # Log everything
    wandb.log({"epoch": config['epochs']+1, "examples_input": examples_input})
    wandb.log({"epoch": config['epochs']+1, "Lower edge": examples_lower_edge})
    wandb.log({"epoch": config['epochs']+1, "Predictions": examples_prediction})
    wandb.log({"epoch": config['epochs']+1, "Upper edge": examples_upper_edge})
    wandb.log({"epoch": config['epochs']+1, "Ground truth": examples_ground_truth})
    wandb.log({"epoch": config['epochs']+1, "Lower length": examples_lower_length})
    wandb.log({"epoch": config['epochs']+1, "Upper length": examples_upper_length})
    """
    # Evaluate the risk and size
    risk, sizes, spearman, stratified_risk, mse = eval_set_metrics(model, val_dataset, config)
    print(f"Risk: {risk}  |  Mean size: {sizes.mean()}  |  Spearman: {spearman}  |  stratified risk: {stratified_risk}  | MSE: {mse}")
    wandb.log({"risk": risk, "mean_size":sizes.mean(), "Spearman":spearman, "Size-Stratified Risk":stratified_risk, "MSE": mse})

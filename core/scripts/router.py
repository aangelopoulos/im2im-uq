import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import wandb
import random
import copy
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision 
import torchvision.transforms as T
import warnings
import yaml

from core.scripts.train import train_net
from core.scripts.eval import get_images, eval_net, get_loss_table, eval_set_metrics 
from core.models.add_uncertainty import add_uncertainty
from core.calibration.calibrate_model import calibrate_model
from core.utils import fix_randomness 
from core.datasets.utils import normalize_dataset 

# Models
from core.models.trunks.unet import UNet

# Datasets
from core.datasets.bsbcm import BSBCMDataset
from core.datasets.fastmri import FastMRIDataset
from core.datasets.temca import TEMCADataset

if __name__ == "__main__":
  # Fix the randomness
  fix_randomness()
  warnings.filterwarnings("ignore")

  print("Entered main method.")
  wandb.init() 
  print("wandb init.")
  # Check if results exist already
  output_dir = wandb.config['output_dir'] 
  results_fname = output_dir + f'/results_' + wandb.config['dataset'] + "_" + wandb.config['uncertainty_type'] + "_" + str(wandb.config['batch_size']) + "_" + str(wandb.config['lr']) + "_" + wandb.config['input_normalization'] + "_" + wandb.config['output_normalization'].replace('.','_') + '.pkl'
  if os.path.exists(results_fname):
    print(f"Results already precomputed and stored in {results_fname}!")
    os._exit(os.EX_OK) 
  else:
    print("Computing the results from scratch!")
  # Otherwise compute results
  curr_method = wandb.config["uncertainty_type"]
  curr_lr = wandb.config["lr"]
  curr_dataset = wandb.config["dataset"]
  wandb.run.name = f"{curr_method}, {curr_dataset}, lr{curr_lr}"
  wandb.run.save()
  params = { key: wandb.config[key] for key in wandb.config.keys() }
  batch_size = wandb.config['batch_size']
  params['batch_size'] = batch_size
  print("wandb save run.")

  # DATASET LOADING
  if wandb.config["dataset"] == "CIFAR10":
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = T.Compose([ T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize ])
    dataset = torchvision.datasets.CIFAR10('/clusterfs/abc/angelopoulos/CIFAR10', download=True, transform=transform)
  elif wandb.config["dataset"] == "bsbcm":
    path = '/home/aa/data/bsbcm'
    dataset = BSBCMDataset(path, num_instances='all', normalize=wandb.config["output_normalization"])
  elif wandb.config["dataset"] == "fastmri":
    path = '~/data/singlecoil_train'
    mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
    dataset = FastMRIDataset(path, normalize_input=wandb.config["input_normalization"], normalize_output = wandb.config["output_normalization"], mask_info=mask_info)
    dataset = normalize_dataset(dataset)
    wandb.config.update(dataset.norm_params)
    params.update(dataset.norm_params)
  elif wandb.config["dataset"] == "temca":
    path = '/clusterfs/fiona/amit/temca_data/'
    dataset = TEMCADataset(path, patch_size=[wandb.config["side_length"], wandb.config["side_length"]], downsampling=[wandb.config["downsampling_factor"],wandb.config["downsampling_factor"]], num_imgs='all', buffer_size=wandb.config["num_buffer"], normalize='01') 
  else:
    raise NotImplementedError 

  # MODEL LOADING
  if wandb.config["dataset"] == "CIFAR10":
    if wandb.config["model"] == "ResNet18":
      trunk = torchvision.models.resnet18(num_classes=wandb.config["num_classes"])
  if wandb.config["model"] == "UNet":
      trunk = UNet(wandb.config["num_inputs"],1)

  # ADD LAST LAYER OF MODEL
  model = add_uncertainty(trunk, params)

  # DATA SPLITTING
  if wandb.config["dataset"] == "temca":
    img_paths = dataset.img_paths
    lengths = np.round(len(img_paths)*np.array(wandb.config["data_split_percentages"])).astype(int)
    lengths[-1] = len(img_paths)-(lengths.sum()-lengths[-1])
    random.shuffle(img_paths)
    train_dataset = copy.deepcopy(dataset)
    calib_dataset = copy.deepcopy(dataset)
    val_dataset = copy.deepcopy(dataset)
    train_dataset.img_paths = img_paths[:lengths[0]]
    calib_dataset.img_paths = img_paths[lengths[0]:(lengths[0]+lengths[1])]
    val_dataset.img_paths = img_paths[(lengths[0]+lengths[1]):(lengths[0]+lengths[1]+lengths[2])]
  else:
    lengths = np.round(len(dataset)*np.array(wandb.config["data_split_percentages"])).astype(int)
    lengths[-1] = len(dataset)-(lengths.sum()-lengths[-1])
    train_dataset, calib_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, lengths.tolist()) 
  
  model = train_net(model,
                    train_dataset,
                    val_dataset,
                    wandb.config['device'],
                    wandb.config['epochs'],
                    batch_size,
                    wandb.config['lr'],
                    wandb.config['load_from_checkpoint'],
                    wandb.config['checkpoint_dir'],
                    wandb.config['checkpoint_every'],
                    wandb.config['validate_every'],
                    params)   

  print("Done training!")
  model.eval()
  with torch.no_grad():
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    #val_loss = eval_net(model,val_loader,wandb.config['device'])
    #print(f"Done validating! Validation Loss: {val_loss}")
    # Save the loss tables for later experiments
    print("Get the validation loss table.") # Doing this first, so I can save it for later experiments.
    val_loss_table = get_loss_table(model,val_dataset,wandb.config)
    print("Calibrate the model.")
    model, calib_loss_table = calibrate_model(model, calib_dataset, params)
    print(f"Model calibrated! lambda hat = {model.lhat}")
    # Save the loss tables
    if output_dir != None:
        try:
            os.makedirs(output_dir,exist_ok=True)
            print('Created output directory')
        except OSError:
            pass
    torch.save(torch.cat((calib_loss_table,val_loss_table),dim=0),output_dir + f'/loss_table_' + wandb.config['dataset'] + "_" + wandb.config['uncertainty_type'] + "_" + str(wandb.config['batch_size']) + "_" + str(wandb.config['lr']) + "_" + wandb.config['input_normalization'] + "_" + wandb.config['output_normalization'].replace('.','_') + '.pth')
    print("Loss table saved!")
    # Get the prediction sets and properly organize them 
    examples_input, examples_lower_edge, examples_prediction, examples_upper_edge, examples_ground_truth, examples_ll, examples_ul, raw_images_dict = get_images(model,
                                                                                                                                                        val_dataset,
                                                                                                                                                         wandb.config['device'],
                                                                                                                                                         list(range(wandb.config['num_validation_images'])),
                                                                                                                                                         params)
    # Log everything
    wandb.log({"epoch": wandb.config['epochs']+1, "examples_input": examples_input})
    wandb.log({"epoch": wandb.config['epochs']+1, "Lower edge": examples_lower_edge})
    wandb.log({"epoch": wandb.config['epochs']+1, "Predictions": examples_prediction})
    wandb.log({"epoch": wandb.config['epochs']+1, "Upper edge": examples_upper_edge})
    wandb.log({"epoch": wandb.config['epochs']+1, "Ground truth": examples_ground_truth})
    wandb.log({"epoch": wandb.config['epochs']+1, "Lower length": examples_ll})
    wandb.log({"epoch": wandb.config['epochs']+1, "Upper length": examples_ul})
    # Get the risk and other metrics 
    print("GET THE METRICS INCLUDING SPATIAL MISCOVERAGE")
    risk, sizes, spearman, stratified_risk, mse, spatial_miscoverage = eval_set_metrics(model, val_dataset, params)
    print("DONE")


    #data = [[label, val] for (label, val) in zip(["Easy","Easy-medium", "Medium-Hard", "Hard"], stratified_risk.numpy())]
    #table = wandb.Table(data=data, columns = ["Difficulty", "Empirical Risk"])
    #wandb.log({"Size-Stratified Risk Barplot" : wandb.plot.bar(table, "Difficulty","Empirical Risk", title="Size-Stratified Risk") })

    print(f"Risk: {risk}  |  Mean size: {sizes.mean()}  |  Spearman: {spearman}  |  Size-stratified risk: {stratified_risk} | MSE: {mse} | Spatial miscoverage: (mu, sigma, min, max) = ({spatial_miscoverage.mean()}, {spatial_miscoverage.std()}, {spatial_miscoverage.min()}, {spatial_miscoverage.max()})")
    wandb.log({"epoch": wandb.config['epochs']+1, "risk": risk, "mean_size":sizes.mean(), "Spearman":spearman, "Size-Stratified Risk":stratified_risk, "mse":mse, "spatial_miscoverage" : spatial_miscoverage})
    
    # Save outputs for later plotting
    print("Saving outputs for plotting")
    if output_dir != None:
        try:
            os.makedirs(output_dir,exist_ok=True)
            print('Created output directory')
        except OSError:
            pass
        results = { "risk": risk, "sizes": sizes, "spearman": spearman, "size-stratified risk": stratified_risk, "mse": mse, "spatial_miscoverage": spatial_miscoverage }
        results.update(raw_images_dict)
        with open(results_fname, 'wb') as handle:
          pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

        print(f'Results saved to file {results_fname}!')

    print(f"Done with {str(params)}")

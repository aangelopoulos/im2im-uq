import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import torch
from core.datasets.CAREDrosophila import CAREDrosophilaDataset
from core.models.trunks.unet import UNet
from core.models.trunks.wnet import WNet
from core.models.add_uncertainty import add_uncertainty
from core.calibration.calibrate_model import calibrate_model 
from core.utils import fix_randomness 
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from core.scripts.train import train_net
from core.scripts.eval import eval_net, eval_risk_size
import yaml
import pdb

import wandb

if __name__ == "__main__":
    fix_randomness()
    wandb.init()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/config.yml') as file:
      config = yaml.safe_load(file)
    path = '/clusterfs/abc/angelopoulos/care/Isotropic_Drosophila/train_data/data_label.npz'
    dataset = CAREDrosophilaDataset(path, num_instances='all', normalize='min-max')
    trunk = UNet(1,1)
    model = add_uncertainty(trunk, config)
    lengths = np.round(len(dataset)*np.array(config["data_split_percentages"])).astype(int)
    lengths[-1] = len(dataset)-(lengths.sum()-lengths[-1])
    train_dataset, calib_dataset, val_dataset = random_split(dataset, lengths.tolist())
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
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    #val_loss = eval_net(model,val_loader,config['device'])
    #print(f"Done validating! Validation Loss: {val_loss}")
    torch.cuda.empty_cache()
    model = calibrate_model(model, calib_dataset, config)
    print(f"Model calibrated! lambda hat = {model.lhat}")
    torch.cuda.empty_cache()
    risk, sizes = eval_risk_size(model, val_dataset, config)
    print(f"Risk: {risk}  |  Mean size: {sizes.mean()}")

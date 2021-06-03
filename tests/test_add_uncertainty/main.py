import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
from core.datasets.CAREDrosophila import CAREDrosophilaDataset
from core.models.trunks.unet import UNet
from core.models.trunks.wnet import WNet
from core.models.add_uncertainty import add_uncertainty
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from core.scripts.train import train_net
import yaml
import pdb



if __name__ == "__main__":
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
    train_net(model,
              train_dataset,
              val_dataset,
              config['device'],
              config['epochs'],
              config['batch_size'],
              config['lr'],
              config['checkpoint_dir'],
              config['checkpoint_every'],
              config['validate_every'])   
    print("Done!")

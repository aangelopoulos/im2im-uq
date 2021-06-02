import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils

class CAREDrosophilaDataset(Dataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    def __init__(self, path, num_instances, normalize=None):
        print('loading dataset from ' + path + '...')
        data = np.load(path) 
        if num_instances == 'all':
          self.x = data['X']
          self.y = data['Y']
        elif num_instances <= data['X'].shape[0]:
          self.x = data['X'][0:num_instances]
          self.y = data['Y'][0:num_instances]
        else:
          print('Dataset only has ' + str(data['X'].shape[0]) + ' instances, please try again')
          exit(0) 
        print('loaded ' + str(self.x.shape[0]) + ' out of '+ str(data['X'].shape[0]) + ' instances')
        del data

        if normalize:
          print('normalizing via ' + normalize + ' normalization ...')
          self.x, self.params = utils.normalize(self.x, type=normalize, per_pixel=False, input_output='input')
          self.y, params_y = utils.normalize(self.y, type=normalize, per_pixel=False, input_output='output')
        self.params.update(params_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx,:,:,:]), torch.from_numpy(self.y[idx,:,:,:])



if __name__ == "__main__":
  path = '/clusterfs/abc/angelopoulos/care/Isotropic_Drosophila/train_data/data_label.npz'
  dataset = CAREDrosophilaDataset(path, num_instances=20, normalize='min-max')
  loader = DataLoader(dataset, batch_size=5, shuffle=True)
  pdb.set_trace()

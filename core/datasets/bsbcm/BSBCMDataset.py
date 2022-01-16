import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils

class BSBCMDataset(Dataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    def __init__(self, path, num_instances, normalize=None):
        print('loading dataset from ' + path + '...')
        x = torch.load(path + '/X.pth')
        y = torch.load(path + '/Y.pth') 
        if num_instances == 'all':
          self.x = x
          self.y = y
        elif num_instances <= x.shape[0]:
          self.x = x[0:num_instances]
          self.y = y[0:num_instances]
        else:
          print('Dataset only has ' + str(x.shape[0]) + ' instances, please try again')
          exit(0) 
        print('loaded ' + str(self.x.shape[0]) + ' out of '+ str(x.shape[0]) + ' instances')
        del x
        del y

        if normalize:
          print('normalizing via ' + normalize + ' normalization ...')
          self.x, self.params = utils.normalize(self.x, type=normalize, per_pixel=False, input_output='input')
          self.y, params_y = utils.normalize(self.y, type=normalize, per_pixel=False, input_output='output')
          self.params.update(params_y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:,:,:], self.y[idx,:,:,:]



if __name__ == "__main__":
  path = '/clusterfs/abc/angelopoulos/bsbcm/'
  dataset = BSBCMDataset(path, num_instances='all', normalize='min-max')
  loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=8)

  for idx, sample in enumerate(loader):
      print(idx)

  pdb.set_trace()
  print("Hi!")

#Useful data utils
import numpy as np
import pickle
from os.path import join
import pdb
from tqdm import tqdm

def normalize(x, type, per_pixel, input_output):
    if type == 'standard':
        # code
        if per_pixel:
            mean_val = x.mean(dim=0)[:, None, :, :]
            std_val = x.std(dim=0)[:, None, :, :]
        else:
            mean_val = x.mean()
            std_val = x.std()
        params = {'mean_'+input_output: mean_val, 'std_'+input_output: std_val}
        x = (x - mean_val) / std_val

    elif type == 'min-max':
        if per_pixel:
            max_val = x.max(dim=0)[0][:, None, :, :]
            min_val = x.min(dim=0)[0][:, None, :, :]
        else:
            max_val = x.max()
            min_val = x.min()
        params = {'max_'+input_output: max_val, 'min_'+input_output: min_val}
        x = (x - min_val) / (max_val - min_val)

    else:
        raise NotImplementedError

    return x, params

def normalize_dataset(dataset):
  param_path = join(dataset.cache_path, 'norm_params.pickle')
  try:
    with open(param_path, 'rb') as handle:
      dataset.norm_params= pickle.load(handle)
    print('normalized with parameters from cache')
  except:
    print('Computing normalization parameters')
    running_max_in = dataset[0][0].max()
    running_min_in = dataset[0][0].min()
    running_max_out = dataset[0][1].max()
    running_min_out = dataset[0][1].min()
    stat_in = RunningStats()
    stat_out = RunningStats()
    for data_point in tqdm(dataset):
      if data_point[0].max() >= running_max_in:
        running_max_in = data_point[0].max()
      if data_point[1].max() >= running_max_out:
        running_max_out = data_point[1].max()
      if data_point[0].min() <= running_min_in:
        running_min_in = data_point[0].min()    
      if data_point[1].min() <= running_min_out:
        running_min_out = data_point[1].min()

      stat_in.push(data_point[0])
      stat_out.push(data_point[1])
         
    dataset.norm_params = {'input_max': running_max_in.item(), 'input_min': running_min_in.item(), 'input_mean': stat_in.mean().item(), 
              'input_std': np.sqrt(stat_in.variance().mean().item()), 'output_max': running_max_out.item(), 'output_min': running_min_out.item(), 
              'output_mean': stat_out.mean().item(), 'output_std': np.sqrt(stat_out.variance().mean()).item()}

    with open(param_path, 'wb') as handle:
      pickle.dump(dataset.norm_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return dataset

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.old_m = self.new_m = x.mean()
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x.mean() - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
        
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0
    
    def standard_deviation(self):
        return np.sqrt(self.variance())

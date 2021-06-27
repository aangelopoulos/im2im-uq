import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import torch
import pickle as pkl
import numpy as np
import random
from tqdm import tqdm


def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def cacheable(func):
    def cache_func(*args):
        cache_dir = str(pathlib.Path(__file__).parent.absolute()) + '/.cache/'
        os.makedirs(cache_dir, exist_ok=True)
        fname = cache_dir + str(func).split(' ')[1] + str(args) + '.pkl'
        if os.path.exists(fname):
            filehandler = open(fname, 'rb')
            return pkl.load(filehandler)
        else:
            filehandler = open(fname, 'wb')
            result = func(*args)
            pkl.dump(result, filehandler)
            return result

    return cache_func


# @cacheable
def normalize_dataset(dataset, type, per_pixel):
    input_norm, input_params = normalize(dataset[:][0], type[0], per_pixel[0], input_output='input')
    output_norm, output_params = normalize(dataset[:][1], type[1], per_pixel[1], input_output='output')
    input_params.update(output_params)
    return torch.utils.data.TensorDataset(input_norm, output_norm), input_params

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

def standard_to_minmax(x,config,output_bool):
  mu = config["output_mean"] if output_bool else config["input_mean"]
  std = config["output_mean"] if output_bool else config["input_mean"]
  lb = config["output_mean"] if output_bool else config["input_mean"]
  ub = config["output_mean"] if output_bool else config["input_mean"]
  x = (((x * std)+mu)-lb)/ub
  return x

def plot_loss(losses, step, path):
    plt.figure()
    plt.plot(np.arange(1, len(losses)+1) * step, losses)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.savefig(path)
    plt.close()

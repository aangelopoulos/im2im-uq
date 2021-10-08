import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import IterableDataset, DataLoader
import pdb
import torch
import torch.nn as nn
import core.datasets.utils as utils
from tqdm import tqdm
import time
from glob import glob
import random
import imageio

mpl.rcParams['figure.dpi'] = 500

class TEMCADataset(IterableDataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    def __init__(self, path, patch_size, downsampling, num_imgs, buffer_size, normalize):
        print('loading dataset from : ' + path  + '...')
        
        # load in flags
        self.path = path
        self.output_size = patch_size #[input_patch_size[0]*upsampling[0], input_patch_size[1]*upsampling[1]]
        self.downsampling = downsampling
        self.buffer_size = buffer_size                
        self.img_index = 0  
        self.normalize = normalize
        self.upsampling_layer = nn.Upsample(size=patch_size)

        # collect all the imgs as filepaths
        self.img_paths = glob(path + '**/*.png')
        random.shuffle(self.img_paths)
        if num_imgs != 'all':
            self.img_paths = self.img_paths[0:num_imgs]
        print('using ' + str(len(self.img_paths)) + ' full images')
        
        # read in the first buffer
        self.patch_buffer = []

    def reset(self):
        self.img_index = 0
        self.patch_buffer = []

    def get_buffer(self):
        if self.img_index + self.buffer_size > len(self.img_paths):
            if len(self.img_paths) -  self.img_index > 0 :
                end_point = len(self.img_paths) - self.img_index
            else:
                self.img_index = -1
                return
        else:
            end_point = self.img_index + self.buffer_size
        
        img_buffer_paths = self.img_paths[self.img_index:end_point]
        for img_path in img_buffer_paths:
            img = imageio.imread(img_path)
            self.extract_patches(imageio.imread(img_path))   
        random.shuffle(self.patch_buffer)
        self.img_index = end_point

    def extract_patches(self, img):
        row_slices = img.shape[0] // self.output_size[0]
        col_slices = img.shape[1] // self.output_size[1]
        for r in range(row_slices):
            for c in range(col_slices):
                  r_start = r*self.output_size[0]
                  r_end = r_start + self.output_size[0] 
                  c_start = c*self.output_size[1]
                  c_end = c_start + self.output_size[1]
                  patch = img[r_start:r_end, c_start:c_end]
                  if patch[np.where(patch==0)].size < 0.85*(patch.shape[0] * patch.shape[1]):
                      self.patch_buffer += [patch]
                

    def __iter__(self):
        while self.img_index != -1:
            if len(self.patch_buffer) == 0:
                self.get_buffer()
            if len(self.patch_buffer) > 0:
                gt = self.patch_buffer.pop().astype(np.float32)#.values.astype(np.float32)
                if self.normalize == '01':
                    gt = gt/255.0
                if self.normalize == '-11':
                    gt = 2*(gt/255.0 - 0.5)
                low_res = torch.from_numpy(gt[None,None,0::self.downsampling[0], 0::self.downsampling[1]])
                low_res = self.upsampling_layer(low_res).squeeze(dim=0)
                high_res = torch.from_numpy(gt[None, :, :])
                yield low_res, high_res
        self.img_index = 0

if __name__ == "__main__":
        
    # Testing the dataset
    dataset = TEMCADataset('/local/amit/temca_data/', patch_size=[2048, 2048], downsampling=[4,4], num_imgs=5, buffer_size=5, normalize='-11') 
    loader = DataLoader(dataset, batch_size=16, drop_last=False, num_workers=0)    
    img = next(iter(dataset))
  
    samp = 0
    for e in range(5):
      print('epoch: ' + str(e))
      for idx, sample in enumerate(loader):
        print(samp)
        samp += sample[1].shape[0]
    print(samp)
    pdb.set_trace()    

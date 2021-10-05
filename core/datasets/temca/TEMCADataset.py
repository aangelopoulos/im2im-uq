import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import IterableDataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils
from tqdm import tqdm
import time
from glob import glob
import random
import imageio

mpl.rcParams['figure.dpi'] = 500

class TEMCADataset(IterableDataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    def __init__(self, path, input_patch_size, upsampling, num_patches, buffer_size=10, normalize=None):
        print('loading dataset from : ' + path  + '...')
        
        # load in flags
        self.path = path
        self.output_size = [input_patch_size[0]*upsampling[0], input_patch_size[1]*upsampling[1]]
        self.upsampling = upsampling
        self.num_patches = num_patches
        self.buffer_size = buffer_size        
        
        # collect all the imgs as filepaths
        self.img_paths = glob(path + '**/*.png')
        print(len(self.img_paths))
        random.shuffle(self.img_paths)
        self.img_index = 0 
 
        #read in the first buffer
        self.patch_buffer = []
        self.get_buffer()
        

    def get_buffer(self):
        if self.img_index + self.buffer_size > len(self.img_paths) - 1:
            if len(self.img_paths) - 1 - self.img_index > 0 :
                end_point = len(self.img_paths) - 1 - img_index + img_index
            else:
                self.img_index = 0
                end_point = self.buffer_size
        else:
            end_point = self.img_index + self.buffer_size
        
        #img_buffer_paths = self.img_paths[self.img_index:end_point]
        img_buffer_paths = self.img_paths
        for img_path in img_buffer_paths:
            print(img_path)
            img = imageio.imread(img_path)
            if img[np.where(img==0)].size > 0.85*(img.shape[0] * img.shape[1]):
                print('NOOOO GET OUUUUUT')
                os.remove(img_path)
            #print(img[np.where(img==0)].size)
            #self.extract_patches(imageio.imread(img_path))   
      

    def extract_patches(self, img):
        row_slices = img.shape[0] // self.output_size[0]
        col_slices = img.shape[1] // self.output_size[1]
        for r in range(row_slices):
            for c in range(col_slices):
                  r_start = r*self.output_size[0]
                  r_end = r_start + self.output_size[0] 
                  c_start = c*self.output_size[1]
                  c_end = c_start + self.output_size[1]
                  self.patch_buffer += [img[r_start:r_end, c_start:c_end]]
        pdb.set_trace()
                


    #def __len__(self):
        
    #    return sum(self.num_patches)

    def __iter__(self, idx):
        gt = self.get_patch(idx).astype(np.float32)#.values.astype(np.float32)
        #if normalize:
        
        return torch.from_numpy(gt[None,0::self.scale[0], 0::self.scale[1]]), torch.from_numpy(gt[None, :, :])        


if __name__ == "__main__":
        
    # Testing the dataset
    dataset = TEMCADataset('/local/amit/temca_data/', input_patch_size=[128, 128], upsampling=[2,2], num_patches='all')
    #pdb.set_trace()
    loader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=2)    
    #print(len(dataset))
    print(len(loader)) 
    prev_time = time.time()
    for idx, sample in enumerate(loader):
        curr_time = time.time()
        print(curr_time-prev_time)
        prev_time = curr_time
        #if d.max() > max_val:
        #max_val = d.max()

    pdb.set_trace()    

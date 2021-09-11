import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils
from core.datasets.fibsem.fibsemtools.src.fibsem_tools.io import read_xarray
from tqdm import tqdm
import time
import xarray as xr

AWS_PATH = 's3://janelia-cosem/'
MAX_GUESS = 5


#list of 4nm res cells 

CELLS_4NM = ["jrc_macrophage-2", "jrc_hela-3", "jrc_hela-2", "jrc_jurkat-1", "jrc_sum159-1", "jrc_hela-4", "jrc_mus-pancreas-1", "jrc_ctl-id8-1", "jrc_fly-acc-calyx-1", "jrc_fly-fsb-1"]

# list of 8nm res cells 
CELLS_8NM = ["jrc_choroid-plexus-2", "jrc_hela-1", "jrc_cos7-11", "jrc_ctl-id8-2", "jrc_hela-h89-2", "jrc_hela-22", "jrc_ctl-id8-5", "jrc_ctl-id8-3", "jrc_hela-h89-1", "jrc_hela-bfa", "jrc_hela-21", "jrc_ctl-id8-4"] 


class FIBSEMDataset(Dataset):
    # path should be absolute, num instances is an int or 'all', normalize can be None, 'standard', or 'min-max'
    def __init__(self, cell_types, input_img_size, scale, hold_out = None, normalize=None):
        print('loading dataset from aws: ' + AWS_PATH  + '...')
        self.size = [input_img_size[0]*scale[0], input_img_size[1]*scale[1]]
        self.cell_datasets = []
        self.num_patches = []
        self.scale = scale
        for cell in cell_types:
            curr = time.time()
            cell_xarray = read_xarray(AWS_PATH + cell + '/' + cell + '.n5/em/fibsem-uint16/s0', storage_options={'anon' : True}).values
            print(time.time() - curr)
            self.cell_datasets += [cell_xarray]        
            #self.num_patches += [ int(np.floor(cell_xarray.x.shape[0]/self.size[1]) * np.floor(cell_xarray.y.shape[0]/self.size[0]) * cell_xarray.z.shape[0]) ]  
        #self.cell_indices = np.cumsum(self.num_patches)
        pdb.set_trace()


    def get_patch(self, idx):
        # locate cell
        which_cell = np.searchsorted(self.cell_indices, idx)
        cell = self.cell_datasets[which_cell]

        # locate patch number within cell
        if which_cell == 0:
            patch_idx = idx
        else:
            patch_idx = idx - self.cell_indices[which_cell - 1]
        
        # locate depth, i.e., z index
        patches_per_depth = int(self.num_patches[which_cell]/cell.z.shape[0])
        which_depth = int(np.floor(patch_idx/patches_per_depth))
        depth_idx = patch_idx - which_depth*patches_per_depth

        # locate row 
        patches_per_row = int(np.floor(cell.x.shape[0]/self.size[1]))
        which_row = int(np.floor(depth_idx/patches_per_row))
        
        # locate column
        which_col = depth_idx - which_row*patches_per_row

        #index patch!
        row_idx = which_row*self.size[0]
        col_idx = which_col*self.size[1]
        #patch = cell[which_depth, row_idx: row_idx + self.size[0], col_idx: col_idx + self.size[1]]
        
        return cell[dict(z=which_depth, y = slice(row_idx, row_idx + self.size[0]), x = slice(col_idx, col_idx + self.size[1]))] 
  

    def __len__(self):
        
        return sum(self.num_patches)

    def __getitem__(self, idx):
        gt = self.get_patch(idx).values.astype(np.float32)
        #if normalize:
        
        return torch.from_numpy(gt[None,0::self.scale[0], 0::self.scale[1]]), torch.from_numpy(gt[None, :, :])        


if __name__ == "__main__":
        
    # Testing the aws data loading
    #uri = 's3://janelia-cosem/jrc_fly-acc-calyx-1/jrc_fly-acc-calyx-1.n5/em/fibsem-uint16/s1' 
    #uri = 's3://janelia-cosem/jrc_macrophage-2/jrc_macrophage-2.n5/em/fibsem-uint16/s0' 
    #result = read_xarray(uri, storage_options={'anon' : True})
    #pic = result[5000,5000:5384,5000:5384].compute().data

    dataset = FIBSEMDataset(cell_types=CELLS_4NM, input_img_size=[128, 128], scale=[2,2])
    
    #loader = DataLoader(dataset, batch_size=32, drop_last=True, num_workers=1)    
    #print(len(dataset))
    #print(len(loader)) 
    #prev_time = time.time()
    #for idx, sample in enumerate(loader):
        #curr_time = time.time()
        #print(curr_time-prev_time)
        #prev_time = curr_time
        #if d.max() > max_val:
        #max_val = d.max()

    pdb.set_trace()    

import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pdb
import torch
import core.datasets.utils as utils
import core.datasets.fastmri.transforms as transforms
import core.datasets.fastmri.subsample as subsample
from pathlib import Path
import random
import h5py
import xml.etree.ElementTree as etree
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

# Much of this code was taken from https://github.com/facebookresearch/fastMRI
class FastMRIDataset(Dataset):
    def __init__(self, path, normalize_input, normalize_output, mask_info, num_volumes=None, slice_sample_period=1):
        # Normalization parameters will be None at first
        self.norm_params = None

        print('loading dataset from ' + path + '...')
        self.challenge = 'singlecoil' 
        self.recons_key = (
            "reconstruction_esc" if self.challenge == "singlecoil" else "reconstruction_rss"
        )
        
        self.cache_path = os.path.join(path, '.cache/')
        os.makedirs(self.cache_path, exist_ok=True) 
        # load the dataset as a list of filenames
        self.examples = []
        
        # gather up volumes
        files = list(Path(path).iterdir())
        random.shuffle(files)
        files = files[0:num_volumes] if (num_volumes and num_volumes < len(files)) else files
        print('Loading ' + str(len(files)) + ' volumes...')
      
        # gather up slices 
        for fname in files:
          if 'cache' in str(fname):
            continue
          metadata, num_slices = self._retrieve_metadata(fname)
          assert(num_slices > slice_sample_period)
          self.examples += [
              (fname, slice_ind, metadata) for slice_ind in range(0,num_slices, slice_sample_period)
          ]
        print('Using ' + str(len(self.examples)) + ' total slices')
        random.shuffle(self.examples)        

        # create sampling mask
        mask_func = subsample.create_mask_for_mask_type( mask_info['type'], mask_info['center_fraction'], mask_info['acceleration'])

        # create a data transform including k-space subsampling mask and normalization
        self.transform = transforms.UnetDataTransform(self.challenge, mask_func=mask_func, use_seed=False)

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, dataslice, metadata = self.examples[idx]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

        if self.normalize_input == 'standard' and self.norm_params != None:
          input_img = (sample[0] - self.norm_params['input_mean'])/self.norm_params['input_std']
        elif self.normalize_input == 'min-max' and self.norm_params != None:
          input_img = (sample[0] - self.norm_params['input_min'])/self.norm_params['input_max']
        else:
          input_img = sample[0]

        if self.normalize_output == 'standard' and self.norm_params != None:
          output_img = (sample[1] - self.norm_params['output_mean'])/self.norm_params['output_std']
        elif self.normalize_output == 'min-max' and self.norm_params != None:
          output_img = (sample[1] - self.norm_params['output_min'])/self.norm_params['output_max']
        else:
          print("No normalization parameters yet.")
          output_img = sample[1]
        

        return (input_img.unsqueeze(0), output_img.unsqueeze(0))

    

if __name__ == "__main__":
  random.seed(1)
  path = '/clusterfs/abc/amit/fastmri/knee/singlecoil_train/'
  mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
  dataset = FastMRIDataset(path, normalize_input='standard', normalize_output = 'min-max', mask_info=mask_info, num_volumes=5)
  utils.normalize_dataset(dataset)
  #loader = DataLoader(dataset, batch_size=5, shuffle=True)
  pdb.set_trace()

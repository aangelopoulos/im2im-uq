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


class FastMRIDataset(Dataset):
    def __init__(self, path, normalize, mask_info, num_volumes=None, slice_sample_period=1):
        
        print('loading dataset from ' + path + '...')
        
        self.cach_path = os.path.join(path, '.cache/')
        
        # load the dataset as a list of filenames
        self.examples = []
        
        # gather up volumes
        files = list(Path(path).iterdir())
        random.shuffle(files)
        files = files[0:num_volumes] if (num_volumes and num_volumes < len(files)) else files
        print('Loading ' + str(len(files)) + ' volumes...')
      
        # gather up slices 
        for fname in files:
          metadata, num_slices = self._retrieve_metadata(fname)
          assert(num_slices > slice_sample_period)
          self.examples += [
              (fname, slice_ind, metadata) for slice_ind in range(0,num_slices, slice_sample_period)
          ]
        print('Using ' + str(len(self.examples)) + ' total slices')
        random.shuffle(self.examples)        

        # create sampling mask
        mask_func = subsample.create_mask_for_mask_type( mask_info['type'], mask_info['center_fraction'], mask_info['acceleration'])

        # create a data transform including k-space subsampling mask and normalizationi
        self.transform = transforms.UnetDataTransform('singlecoil', mask_func=mask_func, use_seed=False)
        #if normalize:
        #  print('normalizing via ' + normalize + ' normalization ...')
        #  self.x, self.params = utils.normalize(self.x, type=normalize, per_pixel=False, input_output='input')
        #  self.y, params_y = utils.normalize(self.y, type=normalize, per_pixel=False, input_output='output')
        #  self.params.update(params_y)

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
            target = hf['reconstruction_esc'][dataslice] if 'reconstruction_esc' in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice)

        return (sample[0], sample[1])

    

if __name__ == "__main__":
  random.seed(1)
  path = '/clusterfs/abc/amit/fastmri/knee/singlecoil_train/'
  mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
  dataset = FastMRIDataset(path, normalize='per_image', mask_info=mask_info)
  loader = DataLoader(dataset, batch_size=5, shuffle=True)
  pdb.set_trace()

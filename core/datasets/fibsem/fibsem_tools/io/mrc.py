import dask.array as da
import numpy as np
from typing import Union, Tuple, List, Sequence
from mrcfile.mrcmemmap import MrcMemmap
from pathlib import Path
from dask.array.core import normalize_chunks
from numpy.typing import ArrayLike
from xarray import DataArray

Pathlike = Union[str, Path]


def access_mrc(path: Pathlike, mode: str, **kwargs):
    # todo: add warning when kwargs are passed to this function
    return MrcMemmap(path, mode=mode)


def mrc_shape_dtype_inference(mem: MrcMemmap) -> Tuple[Tuple[int], str]:
    """
    Infer the shape and datatype of an MrcMemmap array. We cannot use the dtype attribute
    because the MRC2014 specification does not officially support the uint8 datatype,
    but that doesn't stop people from storing uint8 data as int8. This can only be inferred
    by checking if the header.dmax propert exceeds the upper limit of int8 (127).
    """
    shape = mem.data.shape
    dtype = mem.data.dtype
    if dtype == "int8":
        if mem.header.dmax > 127:
            dtype = "uint8"
    return shape, dtype


def mrc_coordinate_inference(mem: MrcMemmap) -> List[DataArray]:
    header = mem.header
    grid_size_angstroms = header.cella
    coords = []
    # round to this many decimal places when calculting the grid spacing, in nm
    grid_spacing_decimals = 2

    if mem.data.flags["C_CONTIGUOUS"]:
        # we reverse the keys from (x,y,z) to (z,y,x) so the order matches
        # numpy indexing order
        keys = reversed(header.cella.dtype.fields.keys())
    else:
        keys = header.cella.dtype.fields.keys()
    for key in keys:
        grid_spacing = np.round((grid_size_angstroms[key] / 10) / header[f"n{key}"], 2)
        axis = np.arange(header[f"n{key}start"], header[f"n{key}"]) * grid_spacing
        coords.append(DataArray(data=axis, dims=(key,), attrs={"units": "nm"}))

    return coords


def mrc_chunk_loader(fname, block_info=None):
    idx = tuple(slice(*idcs) for idcs in block_info[None]["array-location"])
    # block_info[None] contains the output specification
    dtype = block_info[None]["dtype"]
    result = np.array(access_mrc(fname, mode="r").data[idx]).astype(dtype)
    return result


def mrc_to_dask(urlpath: Pathlike, chunks: Union[str, Sequence[int]]):
    """
    Generate a dask array backed by a memory-mapped .mrc file
    """
    with access_mrc(urlpath, mode="r") as mem:
        shape, dtype = mrc_shape_dtype_inference(mem)

    if chunks == "auto":
        _chunks = normalize_chunks((1, *(-1,) * (len(shape) - 1)), shape, dtype=dtype)
    else:
        _chunks = normalize_chunks(chunks, shape, dtype=dtype)

    arr = da.map_blocks(mrc_chunk_loader, urlpath, chunks=_chunks, dtype=dtype)
    return arr

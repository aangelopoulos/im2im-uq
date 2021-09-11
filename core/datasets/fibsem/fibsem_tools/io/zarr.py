from collections import defaultdict
from typing import Dict, List, Tuple, Any, Union, Sequence
from pathlib import Path
from dask import delayed
import dask.array as da
import zarr
import os
from fibsem_tools.io.util import rmtree_parallel, split_path_at_suffix
from toolz import concat
from xarray import DataArray
import numpy as np
from fibsem_tools.io.storage import N5FSStore
from zarr.indexing import BasicIndexer
from distributed import Lock, Client

# default axis order of zarr spatial metadata
# is z,y,x
ZARR_AXES_3D = ["z", "y", "x"]

# default axis order of raw n5 spatial metadata
# is x,y,z
N5_AXES_3D = ZARR_AXES_3D[::-1]


def get_arrays(obj: Any) -> Tuple[zarr.core.Array]:
    result = ()
    if isinstance(obj, zarr.core.Array):
        result = (obj,)
    elif isinstance(obj, zarr.hierarchy.Group):
        if len(tuple(obj.arrays())) > 1:
            names, arrays = zip(*obj.arrays())
            result = tuple(concat(map(get_arrays, arrays)))
    return result


def delete_zbranch(branch, compute=True):
    """
    Delete a branch (group or array) from a zarr container
    """
    if isinstance(branch, zarr.hierarchy.Group):
        return delete_zgroup(branch, compute=compute)
    elif isinstance(branch, zarr.core.Array):
        return delete_zarray(branch, compute=compute)
    else:
        raise TypeError(
            f"The first argument to this function my be a zarr group or array, not {type(branch)}"
        )


def delete_zgroup(zgroup, compute=True):
    """
    Delete all arrays in a zarr group
    """
    if not isinstance(zgroup, zarr.hierarchy.Group):
        raise TypeError(
            f"Cannot use the delete_zgroup function on object of type {type(zgroup)}"
        )

    arrays = get_arrays(zgroup)
    to_delete = delayed([delete_zarray(arr, compute=False) for arr in arrays])

    if compute:
        return to_delete.compute()
    else:
        return to_delete


def delete_zarray(zarray, compute=True):
    """
    Delete a zarr array.
    """

    if not isinstance(zarray, zarr.core.Array):
        raise TypeError(
            f"Cannot use the delete_zarray function on object of type {type(zarray)}"
        )

    path = os.path.join(zarray.store.path, zarray.path)
    store = zarray.store
    branch_depth = None
    if isinstance(store, zarr.N5Store) or isinstance(store, zarr.NestedDirectoryStore):
        branch_depth = 1
    elif isinstance(store, zarr.DirectoryStore):
        branch_depth = 0
    else:
        warnings.warn(
            f"Deferring to the zarr-python implementation for deleting store with type {type(store)}"
        )
        return None

    result = rmtree_parallel(path, branch_depth=branch_depth, compute=compute)
    return result


def same_compressor(arr: zarr.Array, compressor) -> bool:
    """

    Determine if the compressor associated with an array is the same as a different compressor.

    arr: A zarr array
    compressor: a Numcodecs compressor, e.g. GZip(-1)
    return: True or False, depending on whether the zarr array's compressor matches the parameters (name, level) of the
    compressor.
    """
    comp = arr.compressor.compressor_config
    return comp["id"] == compressor.codec_id and comp["level"] == compressor.level


def same_array_props(
    arr: zarr.Array, shape: Tuple[int], dtype: str, compressor: Any, chunks: Tuple[int]
) -> bool:
    """

    Determine if a zarr array has properties that match the input properties.

    arr: A zarr array
    shape: A tuple. This will be compared with arr.shape.
    dtype: A numpy dtype. This will be compared with arr.dtype.
    compressor: A numcodecs compressor, e.g. GZip(-1). This will be compared with the compressor of arr.
    chunks: A tuple. This will be compared with arr.chunks
    return: True if all the properties of arr match the kwargs, False otherwise.
    """
    return (
        (arr.shape == shape)
        & (arr.dtype == dtype)
        & same_compressor(arr, compressor)
        & (arr.chunks == chunks)
    )


def zarr_array_from_dask(arr: Any) -> Any:
    """
    Return the zarr array that was used to create a dask array using `da.from_array(zarr_array)`
    """
    keys = tuple(arr.dask.keys())
    return arr.dask[keys[-1]]


def access_zarr(
    dir_path: Union[str, Path], container_path: Union[str, Path], **kwargs
) -> Any:
    if isinstance(dir_path, Path):
        dir_path = str(dir_path)
    if isinstance(container_path, Path):
        dir_path = str(dir_path)

    attrs = kwargs.pop("attrs", {})

    # zarr is extremely slow to delete existing directories, so we do it ourselves
    if kwargs.get("mode") == "w":
        tmp_kwargs = kwargs.copy()
        tmp_kwargs["mode"] = "a"
        tmp = zarr.open(dir_path, path=str(container_path), **tmp_kwargs)
        # todo: move this logic to methods on the stores themselves
        if isinstance(
            tmp.store, (zarr.N5Store, zarr.DirectoryStore, zarr.NestedDirectoryStore)
        ):
            delete_zbranch(tmp)
    array_or_group = zarr.open(dir_path, path=str(container_path), **kwargs)
    if kwargs.get("mode") != "r" and len(attrs) > 0:
        array_or_group.attrs.update(attrs)
    return array_or_group


def access_n5(
    dir_path: Union[str, Path], container_path: Union[str, Path], **kwargs
) -> Any:
    dir_path = N5FSStore(dir_path, **kwargs.get("storage_options", {}))
    return access_zarr(dir_path, container_path, **kwargs)


def zarr_to_dask(urlpath: str, chunks: Union[str, Sequence[int]], **kwargs):
    store_path, key, _ = split_path_at_suffix(urlpath, (".zarr",))
    arr = access_zarr(store_path, key, mode="r", **kwargs)
    if not hasattr(arr, "shape"):
        raise ValueError(f"{store_path}/{key} is not a zarr array")
    if chunks == "original":
        _chunks = arr.chunks
    else:
        _chunks = chunks
    darr = da.from_array(arr, chunks=_chunks, inline_array=True)
    return darr


def n5_to_dask(urlpath: str, chunks: Union[str, Sequence[int]], **kwargs):
    store_path, key, _ = split_path_at_suffix(urlpath, (".n5",))
    arr = access_n5(store_path, key, mode="r", **kwargs)
    if not hasattr(arr, "shape"):
        raise ValueError(f"{store_path}/{key} is not an n5 array")
    if chunks == "original":
        _chunks = arr.chunks
    else:
        _chunks = chunks
    darr = da.from_array(arr, chunks=_chunks, inline_array=True)
    return darr


def zarr_n5_coordinate_inference(
    shape: Tuple[int, ...], attrs: Dict[str, Any], default_unit: str = "nm"
) -> Tuple[List[DataArray], Dict[str, Any]]:
    output_attrs = attrs.copy()
    input_axes: List[str] = [f"dim_{idx}" for idx in range(len(shape))]
    output_axes: List[str] = input_axes
    units: Dict[str, str] = {ax: default_unit for ax in output_axes}
    scales: Dict[str, float] = {ax: 1.0 for ax in output_axes}
    translates: Dict[str, float] = {ax: 0.0 for ax in output_axes}

    if output_attrs.get("transform"):
        transform_meta = output_attrs.pop("transform")
        input_axes = transform_meta["axes"]
        output_axes = input_axes
        units = dict(zip(output_axes, transform_meta["units"]))
        scales = dict(zip(output_axes, transform_meta["scale"]))
        translates = dict(zip(output_axes, transform_meta["translate"]))

    elif output_attrs.get("pixelResolution") or output_attrs.get("resolution"):
        input_axes = N5_AXES_3D
        output_axes = input_axes[::-1]
        translates = {ax: 0 for ax in output_axes}
        units = {ax: default_unit for ax in output_axes}

        if output_attrs.get("pixelResolution"):
            pixelResolution = output_attrs.pop("pixelResolution")
            scales = dict(zip(input_axes, pixelResolution["dimensions"]))
            units = {ax: pixelResolution["unit"] for ax in input_axes}

        elif output_attrs.get("resolution"):
            _scales = output_attrs.pop("resolution")
            scales = dict(zip(N5_AXES_3D, _scales))

    coords = [
        DataArray(
            translates[ax] + np.arange(shape[idx]) * scales[ax],
            dims=ax,
            attrs={"units": units[ax]},
        )
        for idx, ax in enumerate(output_axes)
    ]

    return coords, output_attrs


def is_n5(array: zarr.core.Array) -> bool:
    if isinstance(array.store, (zarr.N5Store, N5FSStore)):
        return True
    else:
        return False


def get_chunk_keys(array: zarr.core.Array, region=slice(None)) -> Sequence[str]:
    indexer = BasicIndexer(region, array)
    chunk_coords = (idx.chunk_coords for idx in indexer)
    keys = (array._chunk_key(cc) for cc in chunk_coords)
    return keys


class ChunkLock:
    def __init__(self, array: zarr.core.Array, client: Client):
        self._locks = get_chunklock(array, client)
        # from the perspective of a zarr array, metadata has this key regardless of the
        # location on storage. unfortunately, the synchronizer does not get access to the
        # indirection provided by the the store class.

        array_attrs_key = f"{array.path}/.zarray"
        if is_n5(array):
            attrs_path = f"{array.path}/attributes.json"
        else:
            attrs_path = array_attrs_key
        self._locks[array_attrs_key] = Lock(attrs_path, client=client)

    def __getitem__(self, key):
        return self._locks[key]


def get_chunklock(array: zarr.core.Array, client: Client) -> Dict[str, Lock]:
    result = {key: Lock(key, client=client) for key in get_chunk_keys(array)}
    return result


def lock_array(array: zarr.core.Array, client: Client) -> zarr.core.Array:
    lock = ChunkLock(array, client)
    locked_array = zarr.open(
        store=array.store, path=array.path, synchronizer=lock, mode="a"
    )
    return locked_array

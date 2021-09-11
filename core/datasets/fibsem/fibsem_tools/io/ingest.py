import numpy as np
import dask.array as da


def pad_arrays(arrays, constant_values, stack=True):
    """
    Pad arrays with variable axis sizes. A bounding box is calculated across all the arrays and each sub-array is
    padded to fit within the bounding box. This is a light wrapper around dask.array.pad. If `stack` is True,
    the arrays will be combined into a larger array via da.stack.

    Parameters
    ----------
    arrays : An iterable collection of dask arrays
    constant_values : The value to fill when padding images

    constant_values : A number which specifies the fill value / mode to use when padding.

    stack: boolean that determines whether the result is a single dask array (stack=True) or a list of dask arrays (stack=False).

    Returns padded arrays and a list of paddings.
    -------

    """

    shapes = np.array([a.shape for a in arrays])
    bounds = shapes.max(0)
    pad_extent = [
        list(zip([0] * shapes.shape[1], (bounds - np.array(a.shape)).tolist()))
        for a in arrays
    ]

    # pad elements of the first axis differently
    def padfun(array, pad_width, constant_values):
        return np.stack(
            [
                np.pad(a, pad_width, constant_values=cv)
                for a, cv in zip(array, constant_values)
            ]
        )

    # If all the shapes are identical no padding is needed.
    if np.unique(shapes, axis=0).shape[0] == 1:
        padded = arrays
    else:
        padded = [
            a.map_blocks(
                padfun,
                pad_width=pad_extent[ind][1:],
                constant_values=constant_values,
                chunks=tuple(
                    c + p[1] - p[0] for c, p in zip(a.chunksize, pad_extent[ind])
                ),
                dtype=a.dtype,
            )
            for ind, a in enumerate(arrays)
        ]

    return padded, pad_extent


def arrays_from_delayed(args, shapes=None, dtypes=None):
    """

    Parameters
    ----------
    args: a collection of dask.delayed objects representing lazy-loaded arrays.

    shapes: a collection of tuples specifying the shape of each array in args, or None. if None, the first array will be loaded
        using local computation, and the shape of that arrays will be used for all subsequent arrays.

    dtypes: a collection of strings specifying the datatype of each array in args, or None. If None, the first array will be loaded
        using local computation and the dtype of that array will be used for all subsequent arrays.

    Returns a list of dask arrays.
    -------

    """

    if shapes is None or dtypes is None:
        sample = args[0].compute(scheduler="threads")
        if shapes is None:
            shapes = (sample.shape,) * len(args)
        if dtypes is None:
            dtypes = (sample.dtype,) * len(args)

    assert len(shapes) == len(args) and len(dtypes) == len(args)

    arrays = [
        da.from_delayed(args[ind], shape=shapes[ind], dtype=dtypes[ind])
        for ind in range(len(args))
    ]
    return arrays

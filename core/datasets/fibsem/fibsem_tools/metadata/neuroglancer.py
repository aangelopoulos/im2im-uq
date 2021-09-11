from pydantic import BaseModel, PositiveInt
from typing import Sequence
from xarray import DataArray
from .transform import SpatialTransform
import numpy as np


class PixelResolution(BaseModel):
    """
    PixelResolution attribute used by the Saalfeld lab. The dimensions attribute contains a list of scales that define the
    grid spacing of the data, in F-order.
    """

    dimensions: Sequence[float]
    unit: str


# todo: validate argument lengths
class NeuroglancerN5GroupMetadata(BaseModel):
    """
    Metadata to enable displaying an N5 group containing several datasets as a multiresolution dataset in neuroglancer.
    see https://github.com/google/neuroglancer/issues/176#issuecomment-553027775
    Axis properties will be indexed in the opposite order of C-contiguous axis indexing.
    """

    axes: Sequence[str]
    units: Sequence[str]
    scales: Sequence[Sequence[PositiveInt]]
    pixelResolution: PixelResolution

    @classmethod
    def fromDataArrays(
        cls, dataarrays: Sequence[DataArray]
    ) -> "NeuroglancerN5GroupMetadata":
        """
        Create neuroglancer-compatibled N5 metadata from a collection of DataArrays.

        Parameters
        ----------

        dataarrays : list or tuple of xarray.DataArray
            The collection of arrays from which to generate multiscale metadata. These arrays are assumed to share the same `dims` attributes, albeit with varying `coords`.

        Returns
        -------

        NeuroglancerN5GroupMetadata
        """
        transforms = [
            SpatialTransform.fromDataArray(array, reverse_axes=True)
            for array in dataarrays
        ]
        pixelresolution = PixelResolution(
            dimensions=transforms[0].scale, unit=transforms[0].units[0]
        )
        scales: List[List[int]] = [
            np.round(np.divide(t.scale, transforms[0].scale)).astype("int").tolist()
            for t in transforms
        ]
        return cls(
            axes=transforms[0].axes,
            units=transforms[0].units,
            scales=scales,
            pixelResolution=pixelresolution,
        )

from pydantic import BaseModel, root_validator
from typing import Sequence, Union, Dict
from xarray import DataArray


class SpatialTransform(BaseModel):
    """
    Representation of an N-dimensional scaling + translation transform for labelled axes with units.
    """

    axes: Sequence[str]
    units: Sequence[str]
    translate: Sequence[float]
    scale: Sequence[float]

    @root_validator
    def validate_argument_length(
        cls, values: Dict[str, Union[Sequence[str], Sequence[float]]]
    ):
        scale = values.get("scale")
        axes = values.get("axes")
        units = values.get("units")
        translate = values.get("translate")
        if not len(axes) == len(units) == len(translate) == len(scale):
            raise ValueError(
                f"The length of all arguments must match. len(axes) = {len(axes)},  len(units) = {len(units)}, len(translate) = {len(translate)}, len(scale) = {len(scale)}"
            )
        return values

    @classmethod
    def fromDataArray(
        cls, dataarray: DataArray, reverse_axes: bool = False
    ) -> "SpatialTransform":
        """
        Generate a spatial transform from a DataArray.

        Parameters
        ----------

        dataarray: DataArray

        reverse_axes: boolean, default=False
            If True, the order of the `axes` in the spatial transform will be reversed relative to the order of the dimensions of `dataarray`.

        Returns
        -------

        SpatialTransform

        """

        orderer = slice(None)
        if reverse_axes:
            orderer = slice(-1, None, -1)
        axes = [str(d) for d in dataarray.dims[orderer]]
        units = [dataarray.coords[ax].attrs.get("units") for ax in axes]
        translate = [float(dataarray.coords[ax][0]) for ax in axes]
        scale = []
        for ax in axes:
            if len(dataarray.coords[ax]) > 1:
                scale_estimate = abs(
                    float(dataarray.coords[ax][1]) - float(dataarray.coords[ax][0])
                )
            else:
                raise ValueError(
                    f"Cannot infer scale parameter along dimension {ax} with length {len(dataarray.coords[ax])}"
                )
            scale.append(scale_estimate)

        return cls(axes=axes, units=units, translate=translate, scale=scale)

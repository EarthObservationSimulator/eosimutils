"""
.. module:: eosimutils.utils
    :synopsis: Module to provide utility functions and classes.

"""

from typing import Type, Union

from eosimutils.state import (
    CartesianState,
    GeographicPosition,
    Cartesian3DPosition,
)
from eosimutils.trajectory import StateSeries, PositionSeries


def convert_object(
    source_obj: Union[
        Cartesian3DPosition,
        CartesianState,
        GeographicPosition,
        StateSeries,
        PositionSeries,
    ],
    target_type: Type,
) -> Union[
    Cartesian3DPosition,
    CartesianState,
    GeographicPosition,
    StateSeries,
    PositionSeries,
]:
    """
    Convert an object to a different type if possible.

    Args:
        source_obj (Union[Cartesian3DPosition, CartesianState,
                        GeographicPosition, StateSeries, PositionSeries]): The object to convert.
        target_type (Type): The target type to convert the object to.

    Returns:
        Union[Cartesian3DPosition, CartesianState,
              GeographicPosition, StateSeries, PositionSeries]: The converted object.
    """
    if not isinstance(
        source_obj,
        (
            Cartesian3DPosition,
            CartesianState,
            GeographicPosition,
            StateSeries,
            PositionSeries,
        ),
    ):
        raise TypeError(
            "Only following source object types are supported: Cartesian3DPosition, "
            "CartesianState, GeographicPosition, StateSeries, or PositionSeries."
        )

    source_type = type(source_obj)

    if target_type == source_type:
        return source_obj

    if target_type == Cartesian3DPosition:
        if source_type == GeographicPosition:
            return source_obj.to_cartesian3d_position()
        elif source_type == CartesianState:
            return source_obj.position
        else:
            raise NotImplementedError(
                f"Conversion from {source_type} to Cartesian3DPosition is not implemented."
            )

    if target_type == StateSeries:
        if source_type == PositionSeries:
            return StateSeries.from_position_series(source_obj)
        else:
            raise NotImplementedError(
                f"Conversion from {source_type} to StateSeries is not implemented."
            )

    if target_type == PositionSeries:
        if source_type == StateSeries:
            return PositionSeries.from_state_series(source_obj)
        else:
            raise NotImplementedError(
                f"Conversion from {source_type} to PositionSeries is not implemented."
            )

    raise ValueError(f"Unsupported target type: {target_type}")

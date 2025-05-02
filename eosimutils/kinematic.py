"""
.. module:: eosimutils.kinematic
   :synopsis: Collection of kinematic transformation methods.

Collection of classes and functions for handling kinematic transformations.

Each function has two versions: one which takes `eosimutils` objects
and another which takes numpy arrays (wherever possible).
The numpy array version is used internally by the `eosimutils` version.
"""

import numpy as np
from typing import Optional, Union

import spiceypy as spice

from .frames import ReferenceFrame
from .time import AbsoluteDate
from .state import Cartesian3DPosition, CartesianState


def _transform_position_vector(
    from_frame: Optional[Union[ReferenceFrame, str]],
    to_frame: Optional[Union[ReferenceFrame, str]],
    position_vector: np.ndarray,
    et: Optional[float] = None,
) -> np.ndarray:
    """Transform a position vector from one reference frame to another.

    Args:
        from_frame (Union[ReferenceFrame, str, None]): The reference frame of the input position.
        to_frame (Union[ReferenceFrame, str, None]): The reference frame to transform to.
        position_vector (np.ndarray): The position vector to transform.
        et (float): Time in seconds in TDB time-scale since J2000
                        (in SPICE: Ephemeris Time (ET)).

    Returns:
        Cartesian3DPosition: The transformed position vector in the target frame.
    """
    if from_frame == to_frame:
        # No transformation needed, return the same position
        return position_vector

    if from_frame == ReferenceFrame.ICRF_EC and to_frame == ReferenceFrame.ITRF:
        # Transform from ICRF_EC to ITRF
        if et is not None:
            rot_matrix = spice.pxform("J2000", "ITRF93", et)
        else:
            raise ValueError(
                "Ephemeris time (ET) must be provided for transformation."
            )
    elif (
        from_frame == ReferenceFrame.ITRF and to_frame == ReferenceFrame.ICRF_EC
    ):
        # Transform from ITRF to ICRF_EC
        if et is not None:
            rot_matrix = spice.pxform("ITRF93", "J2000", et)
        else:
            raise ValueError(
                "Ephemeris time (ET) must be provided for transformation."
            )
    else:
        raise NotImplementedError(
            f"Transformation from {from_frame} to {to_frame} is not implemented."
        )
    return rot_matrix @ position_vector


def transform_position(
    from_frame: Optional[Union[ReferenceFrame, str]],
    to_frame: Optional[Union[ReferenceFrame, str]],
    position: Cartesian3DPosition,
    time: Optional[AbsoluteDate] = None,
) -> Cartesian3DPosition:
    """Transform a position vector from one reference frame to another.

    Args:
        from_frame (Union[ReferenceFrame, str, None]): The reference frame of the input position.
        to_frame (Union[ReferenceFrame, str, None]): The reference frame to transform to.
        position (Cartesian3DPosition): The position vector to transform.
        time (Optional[AbsoluteDate]): Time.

    Returns:
        Cartesian3DPosition: The transformed position vector in the target frame.
    """
    transformed_position = _transform_position_vector(
        from_frame=from_frame,
        to_frame=to_frame,
        position_vector=position.to_numpy(),
        et=time.to_spice_ephemeris_time(),
    )

    return Cartesian3DPosition.from_array(transformed_position, to_frame)


def _transform_state(
    from_frame: Optional[Union[ReferenceFrame, str]],
    to_frame: Optional[Union[ReferenceFrame, str]],
    state: np.ndarray,
    et: Optional[float] = None,
) -> np.ndarray:
    """Transform a state vector from one reference frame to another.

    Args:
        from_frame (Union[ReferenceFrame, str, None]): The reference frame of the input state.
        to_frame (Union[ReferenceFrame, str, None]): The reference frame to transform to.
        state (np.ndarray): The state vector to transform.
        et (float): Time in seconds in TDB time-scale since J2000
                        (in SPICE: Ephemeris Time (ET)).
    Returns:
        np.ndarray: The transformed state vector in the target frame.
    """
    if from_frame == to_frame:
        # No transformation needed, return the same state
        return state

    if from_frame == ReferenceFrame.ICRF_EC and to_frame == ReferenceFrame.ITRF:
        state_matrix = spice.sxform("J2000", "ITRF93", et)
    elif (
        from_frame == ReferenceFrame.ITRF and to_frame == ReferenceFrame.ICRF_EC
    ):
        state_matrix = spice.sxform("ITRF93", "J2000", et)
    else:
        raise NotImplementedError(
            f"Transformation from {from_frame} to {to_frame} is not implemented."
        )
    return state_matrix @ state


def transform_state(
    from_frame: Optional[Union[ReferenceFrame, str]],
    to_frame: Optional[Union[ReferenceFrame, str]],
    state: CartesianState,
    time: Optional[AbsoluteDate] = None,
) -> CartesianState:
    """Transform a state vector from one reference frame to another.

    Args:
        from_frame (Union[ReferenceFrame, str, None]): The reference frame of the input state.
        to_frame (Union[ReferenceFrame, str, None]): The reference frame to transform to.
        state (CartesianState): The state vector to transform.
        time (Optional[AbsoluteDate]): Time.

    Returns:
        CartesianState: The transformed state vector in the target frame.
    """
    transformed_state = _transform_state(
        from_frame=from_frame,
        to_frame=to_frame,
        state=state.to_numpy(),
        et=time.to_spice_ephemeris_time(),
    )

    return CartesianState.from_array(transformed_state, time, to_frame)

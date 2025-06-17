"""Utility functions to help with the tests."""

import numpy as np
from typing import Union

from eosimutils.thirdpartyutils import astropy_transform
from eosimutils.state import Cartesian3DPosition, CartesianState
from eosimutils.time import AbsoluteDate
from eosimutils.base import ReferenceFrame
from eosimutils.orientation import SpiceOrientation


def validate_transform_state_with_astropy(
    from_frame: Union[str, ReferenceFrame],
    to_frame: Union[str, ReferenceFrame],
    state: CartesianState,
) -> bool:
    """
    Validate the `eosimutils.orientation.SpiceOrientation.transform_state` 
    function using astropy_transform.

    Args:
        from_frame (Union[str, ReferenceFrame]): The reference frame of the input state.
        to_frame (Union[str, ReferenceFrame]): The reference frame to transform to.
        state (CartesianState): The state vector (position and velocity) to transform.

    Returns:
        bool: True if the transform_state output matches astropy_transform for both
            position and velocity, False otherwise.
    """
    time = state.time
    # Use eosimutils transform_state to get the transformed state
    test_orientation = SpiceOrientation(
        from_frame=from_frame,
        to_frame=to_frame,
    )
    transform_state = test_orientation.transform_state
    transformed_state = transform_state(
        state=state,
    )

    # Transform using astropy
    # Get the frame names in a format accepted by astropy
    if from_frame == "GCRF" or from_frame == ReferenceFrame.get("ICRF_EC"):
        from_frame_str = "GCRS"
    elif from_frame == "ITRF" or from_frame == ReferenceFrame.get("ITRF"):
        from_frame_str = "ITRS"
    else:
        raise ValueError(f"Unsupported from_frame: {from_frame}")

    if to_frame == "GCRF" or to_frame == ReferenceFrame.get("ICRF_EC"):
        to_frame_str = "GCRS"
    elif to_frame == "ITRF" or to_frame == ReferenceFrame.get("ITRF"):
        to_frame_str = "ITRS"
    else:
        raise ValueError(f"Unsupported to_frame: {to_frame}")

    # Use astropy_transform to get the validation data
    transformed_position_astropy, transformed_velocity_astropy = (
        astropy_transform(
            position=state.position.to_numpy(),
            velocity=state.velocity.to_numpy(),
            from_frame=from_frame_str,
            to_frame=to_frame_str,
            timestamp=time.to_dict()["calendar_date"],
        )
    )

    # Compare the results for position and velocity
    position_matches = np.allclose(
        transformed_state.position.to_numpy(),
        transformed_position_astropy,
        atol=1e-6,
    )
    velocity_matches = np.allclose(
        transformed_state.velocity.to_numpy(),
        transformed_velocity_astropy,
        atol=1e-6,
    )
    return position_matches and velocity_matches


def validate_transform_position_with_astropy(
    from_frame: Union[str, ReferenceFrame],
    to_frame: Union[str, ReferenceFrame],
    position: Cartesian3DPosition,
    time: AbsoluteDate,
) -> bool:
    """
    Validate the `eosimutils.orientation.SpiceOrientation.transform_position` 
    function using astropy_transform..

    Args:
        from_frame (Union[str, ReferenceFrame]): The reference frame of the input position.
        to_frame (Union[str, ReferenceFrame]): The reference frame to transform to.
        position (Cartesian3DPosition): The position vector to transform.
        time (AbsoluteDate): The time of the transformation.

    Returns:
        bool: True if the transform_position output matches astropy_transform, False otherwise.
    """
    # Use eosimutils transform_position to get the transformed position
    test_orientation = SpiceOrientation(
        from_frame=from_frame,
        to_frame=to_frame,
    )
    transform_position = test_orientation.transform_position
    transformed_position = transform_position(
        position=position,
        t=time,
    )

    # Transform using astropy
    # get the frame names in format accepted by astropy
    if from_frame == "GCRF" or from_frame == ReferenceFrame.get(
        "ICRF_EC"
    ):  # Orientation of ICRF ~ orientation of GCRF
        from_frame_str = "GCRS"
    elif from_frame == "ITRF" or from_frame == ReferenceFrame.get("ITRF"):
        from_frame_str = "ITRS"
    else:
        raise ValueError(f"Unsupported from_frame: {from_frame}")

    if to_frame == "GCRF" or to_frame == ReferenceFrame.get(
        "ICRF_EC"
    ):  # Orientation of ICRF ~ orientation of GCRF
        to_frame_str = "GCRS"
    elif to_frame == "ITRF" or to_frame == ReferenceFrame.get("ITRF"):
        to_frame_str = "ITRS"
    else:
        raise ValueError(f"Unsupported to_frame: {to_frame}")

    transformed_position_astropy, _ = astropy_transform(
        position=position.to_numpy(),
        velocity=[
            0.0,
            0.0,
            0.0,
        ],  # Velocity is not needed for position validation
        from_frame=from_frame_str,
        to_frame=to_frame_str,
        timestamp=time.to_dict()["calendar_date"],
    )

    # Compare the results
    return np.allclose(
        transformed_position.to_numpy(),
        transformed_position_astropy,
        atol=1e-6,
    )

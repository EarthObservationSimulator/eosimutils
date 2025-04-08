"""
.. module:: eosimutils.thirdpartyutils
   :synopsis: Collection of miscellaneous utility functions based on third-party libraries.

Collection of miscellaneous utility functions based on third-party libraries.
The functions are designed to be independent of the main library and can be used as standalone utilities.
They are also used for testing purposes to validate the functionality of the main library.
"""

from typing import List, Tuple, Any, Union
import numpy as np

from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from astropy import units as u

def astropy_transform(
    position: Union[List[float], np.ndarray],
    velocity: Union[List[float], np.ndarray],
    from_frame: str,
    to_frame: str,
    timestamp: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform satellite position and velocity between reference frames using Astropy.

    Args:
        position (Union[List[float], np.ndarray]): Satellite position [x, y, z] in the input frame (in kilometers).
        velocity (Union[List[float], np.ndarray]): Satellite velocity [vx, vy, vz] in the input frame (in kilometers per second).
        from_frame (str): The reference frame of the input position and velocity (e.g., 'ICRS', 'GCRS', 'ITRS').
        to_frame (str): The reference frame to transform to (e.g., 'ICRS', 'ITRS', 'GCRS').
        timestamp (str): A UTC string representing the time of the observation (e.g., "2025-03-17T12:00:00").

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Transformed position [x, y, z] in the target frame (in kilometers).
            - Transformed velocity [vx, vy, vz] in the target frame (in kilometers per second).

    Raises:
        ValueError: If `from_frame` or `to_frame` is not supported.
    """
    # Convert the time to an astropy Time object
    t = Time(timestamp, scale="utc")

    # Create the position and velocity as astropy CartesianRepresentation objects
    input_pos = CartesianRepresentation(position * u.km)
    input_vel = CartesianDifferential(velocity * u.km / u.s)

    # Create the full state (position + velocity)
    input_state = input_pos.with_differentials(input_vel)

    # Dynamically create the input frame
    if from_frame == "GCRS": 
        # GCRF (geocentric) ~ ICRF (solar system barycenter) in orientation
        input_frame = GCRS(input_state, obstime=t)
    elif from_frame == "ITRS":
        input_frame = ITRS(input_state, obstime=t)
    else:
        raise ValueError(f"Unsupported from_frame: {from_frame}")

    # Transform to the target frame
    if to_frame == "GCRS":
        target_frame = GCRS(obstime=t)
    elif to_frame == "ITRS":
        target_frame = ITRS(obstime=t)
    else:
        raise ValueError(f"Unsupported to_frame: {to_frame}")

    transformed = input_frame.transform_to(target_frame)

    # Extract the transformed position and velocity
    transformed_position = transformed.cartesian.xyz.to(u.km).value
    transformed_velocity = transformed.velocity.d_xyz.to(u.km / u.s).value

    return transformed_position, transformed_velocity
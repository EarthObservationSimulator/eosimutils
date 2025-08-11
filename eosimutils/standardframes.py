"""
.. module:: eosimutils.standardframes
   :synopsis: Functions to compute commonly used reference frames..
"""

from scipy.spatial.transform import Rotation as Scipy_Rotation
import numpy as np

from .base import ReferenceFrame
from .orientation import OrientationSeries
from .trajectory import StateSeries, PositionSeries


def get_lvlh(
    state: StateSeries, lvlh_frame: ReferenceFrame
) -> tuple["OrientationSeries", "PositionSeries"]:
    """
    Compute an LVLH OrientationSeries and PositionSeries from a satellite's StateSeries.

    The axes are constructed as follows:

        - Z-axis (Local Vertical): negative unit position vector (-r/|r|).
        - Y-axis (Cross-track): negative unit angular momentum vector (-h/|h|, where h = r × v).
        - X-axis (Local Horizontal): cross product of Y and Z axes (x = y × z).

    Reference: https://sanaregistry.org/r/orbit_relative_reference_frames/ (LVLH_ROTATING)

    Args:
        state (StateSeries): Trajectory in an inertial frame.
        lvlh_frame (ReferenceFrame): The ReferenceFrame object for newly created LVLH frame.

    Returns:
        tuple: (OrientationSeries, PositionSeries)
            - OrientationSeries: LVLH OrientationSeries with LVLH as from_frame and state.frame as
            to_frame.
            - PositionSeries: Position of the LVLH frame origin relative to the planet center,
            expressed in inertial coordinates.

    Raises:
        ValueError: If the state frame is not an inertial frame.
    """
    # Only support ICRF_EC as inertial
    # if state.frame != ReferenceFrame.get("ICRF_EC"):
    #     raise ValueError(
    #         f"get_LVLH only defined for inertial frames, got {state.frame}"
    #     )

    pos = state.data[0]  # shape (N,3)
    vel = state.data[1]  # shape (N,3)

    r_mats = []
    # Form the matrix where each column is a basis vector
    # of the LVLH frame, expressed in inertial coordinates.
    # This is the matrix which transforms vectors from the LVLH to the inertial frame.
    for r, v in zip(pos, vel):
        r_norm = np.linalg.norm(r)
        z_hat = -r / r_norm  # Local vertical: -r/|r|
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)
        y_hat = -h / h_norm  # Cross-track:-h/|h|
        x_hat = np.cross(y_hat, z_hat)  # Local horizontal: y × z
        r_mats.append(np.column_stack([x_hat, y_hat, z_hat]))

    rotations = Scipy_Rotation.from_matrix(np.array(r_mats))
    orientation = OrientationSeries(
        time=state.time,
        rotations=rotations,
        from_frame=lvlh_frame,
        to_frame=state.frame,
    )

    # Position of LVLH origin (spacecraft) relative to planet center, in inertial frame
    position_series = PositionSeries(
        state.time,
        pos,
        state.frame,
    )

    return orientation, position_series
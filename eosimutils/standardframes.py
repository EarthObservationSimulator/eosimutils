"""
.. module:: eosimutils.standardframes
   :synopsis: Functions to compute commonly used reference frames.
"""

from typing import Dict, Any, Type, Callable
from scipy.spatial.transform import Rotation as Scipy_Rotation
import numpy as np

from .base import ReferenceFrame, EnumBase
from .orientation import OrientationSeries
from .trajectory import StateSeries, PositionSeries


class StandardFrameType(EnumBase):
    """Standard supported frame type definitions."""

    LVLH_TYPE_1 = "LVLH_TYPE_1"  # LVLH_INERTIAL_Z_NEGATIVE_POSITION_Y_NEGATIVE_ANGULAR_MOMENTUM_X_CROSS_RIGHT_HAND # pylint: disable=line-too-long


class StandardFrameHandlerFactory:
    """Factory class to register and handle standard frames.
    The word "Handler" is used to avoid confusion with ReferenceFrame class.
    """

    # Class-level registry for standard frame types
    _registry: Dict[str, Type] = {}

    @classmethod
    def register_type(cls, type_name: str) -> Callable[[Type], Type]:
        """
        Decorator to register an standard frame class under a type name.
        """

        def decorator(standard_frame_class: Type) -> Type:
            cls._registry[type_name] = standard_frame_class
            return standard_frame_class

        return decorator

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> object:
        """
        Retrieves an instance of the appropriate standard frame class based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing standard frame specifications.
                Must include a valid frame type in the "frame_type" key.

        Returns:
            object: An instance of the appropriate standard frame class initialized with
                    the given specifications.

        Raises:
            KeyError: If the "frame_type" key is missing in the specifications dictionary.
            ValueError: If the specified frame type is not registered.
        """
        frame_type_str = specs.get("frame_type")
        if frame_type_str is None:
            raise KeyError(
                'Frame type key "frame_type" not found in specifications dictionary.'
            )
        frame_class = cls._registry.get(frame_type_str)
        if not frame_class:
            raise ValueError(
                f'Frame type "{frame_type_str}" is not registered.'
            )
        return frame_class.from_dict(specs)


@StandardFrameHandlerFactory.register_type(StandardFrameType.LVLH_TYPE_1.value)
class LVLHType1FrameHandler:
    """
    Class to register and compute LVLH (Local Vertical Local Horizontal) type-1 reference frame.
    The instance variable `frame` holds the ReferenceFrame object for the LVLH frame.

    See `eosimutils.standardframes.get_lvlh` function for details on the frame definition.
    """

    def __init__(self, frame_name: str):
        """
        Initializes an LVLHType1FrameHandler instance.
        The frame is registered (if not already) with the
        ReferenceFrame class upon initialization.

        Args:
            frame_name (str): Name of the LVLH frame.
        """
        if ReferenceFrame.get(frame_name) is None:
            ReferenceFrame.add(frame_name)
        self.frame: ReferenceFrame = ReferenceFrame.get(frame_name)

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "LVLHType1FrameHandler":
        """
        Creates an LVLHType1FrameHandler instance and registers the frame
        from a specifications dictionary.

        Args:
            specs (Dict[str, Any]): A dictionary containing LVLH frame specifications.
                Expected keys:
                - "name" (str): Name of the LVLH frame to be created.

        Returns:
            LVLHType1FrameHandler: An instance of the LVLHType1FrameHandler class
                                initialized with the given specifications.

        Raises:
            KeyError: If required keys are missing in the specifications dictionary.
        """
        name = specs.get("name")
        if name is None:
            raise KeyError('Key "name" not found in specifications dictionary.')
        return cls(name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the LVLHType1FrameHandler instance to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the LVLH frame specifications.
                Keys:
                - "frame_type" (str): The type of the frame ("LVLH_TYPE_1").
                - "name" (str): Name of the LVLH frame.
        """
        return {
            "frame_type": StandardFrameType.LVLH_TYPE_1.value,
            "name": self.frame.to_string(),
        }

    def get_frame(self) -> ReferenceFrame:
        """
        Returns the ReferenceFrame object for the LVLH frame.

        Returns:
            ReferenceFrame: The ReferenceFrame object for the LVLH frame.
        """
        return self.frame

    def get_transform(
        self, state: StateSeries
    ) -> tuple["OrientationSeries", "PositionSeries"]:
        """
        Compute an LVLH OrientationSeries and PositionSeries from a
        satellite's StateSeries (in inertial frame).

        The axes are constructed as follows:

            - Z-axis (Local Vertical): negative unit position vector (-r/|r|).
            - Y-axis (Cross-track): negative unit angular momentum vector (-h/|h|, where h = r × v).
            - X-axis (Local Horizontal): cross product of Y and Z axes (x = y × z).

                    Z (Local Vertical)
                ^
                |
                |   -r/|r|
                |
                +-------> X (Local Horizontal)
                /
                /
            Y (Cross-track)
            -h/|h|; h = r × v

        Reference: https://sanaregistry.org/r/orbit_relative_reference_frames/ (LVLH_ROTATING)

        Args:
            state (StateSeries): Trajectory in an inertial frame.

        Returns:
            tuple: (OrientationSeries, PositionSeries)
                - OrientationSeries: LVLH OrientationSeries with LVLH as `from_frame` 
                                        and self.frame as `to_frame`.
                - PositionSeries: Position of the LVLH frame origin relative to the planet center,
                expressed in inertial coordinates.

        Raises:
            ValueError: If the state frame is not an inertial frame.
        """

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
            from_frame=self.frame,
            to_frame=state.frame,
        )

        # Position of LVLH origin (spacecraft) relative to planet center, in inertial frame
        position_series = PositionSeries(
            state.time,
            pos,
            state.frame,
        )

        return orientation, position_series

    def __eq__(self, value):
        if not isinstance(value, LVLHType1FrameHandler):  # type is checked
            return False
        return self.frame == value.frame

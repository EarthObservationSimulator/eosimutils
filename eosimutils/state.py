"""
.. module:: eosimutils.state
   :synopsis: State vector information.

Collection of classes and functions for handling state vector information.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional

from skyfield.positionlib import build_position as skyfield_build_position
from skyfield.constants import AU_KM as Skyfield_AU_KM
from skyfield.api import wgs84 as skyfield_wgs84

from .base import ReferenceFrame
from .time import AbsoluteDate


class Cartesian3DPosition:
    """Handles 3D position information. 
       Internally the position is stored in kilometers.
    """

    def __init__(
        self, x: float, y: float, z: float, frame: Optional[ReferenceFrame]
    ) -> None:
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            raise ValueError("x, y, and z must be numeric values.")
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError("frame must be a ReferenceFrame object or None.")
        self.coords = np.array([x, y, z])
        self.frame = frame

    @staticmethod
    def from_array(
        array_in: Union[List[float], np.ndarray, Tuple[float, float, float]],
        frame: Optional[Union[ReferenceFrame, str, None]] = None,
    ) -> "Cartesian3DPosition":
        """Construct a Cartesian3DPosition object from a list, tuple, or NumPy array.

        Args:
            array_in (Union[List[float], np.ndarray, Tuple[float, float, float]]):
                Position coordinates in kilometers.
            frame (Union[ReferenceFrame, str, None]): Reference-frame.

        Returns:
            Cartesian3DPosition: Cartesian3DPosition object.
        """
        if isinstance(array_in, np.ndarray):
            array_in = array_in.tolist()  # Convert NumPy array to list
        elif isinstance(array_in, tuple):
            array_in = list(array_in)  # Convert tuple to list

        if len(array_in) != 3:
            raise ValueError("The input must contain exactly 3 elements.")
        if not all(isinstance(coord, (int, float)) for coord in array_in):
            raise ValueError(
                "All elements in the input must be numeric values."
            )
        if isinstance(frame, str):
            frame = ReferenceFrame.get(frame)
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError(
                "frame must be a ReferenceFrame object, a valid string, or None."
            )
        return Cartesian3DPosition(array_in[0], array_in[1], array_in[2], frame)

    def to_numpy(self) -> np.ndarray:
        """Convert the Cartesian3DPosition object to a NumPy array.

        Returns:
            np.ndarray: Position coordinates in kilometers.
        """
        return self.coords

    def to_list(self) -> List[float]:
        """Convert the Cartesian3DPosition object to a list.

        Returns:
            List[float]: List with the position coordinates in kilometers.
        """
        return self.coords.tolist()

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Cartesian3DPosition":
        """Construct a Cartesian3DPosition object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the position information.
                The dictionary should contain the following key-value
                pairs:
                - "x" (float): The x-coordinate in kilometers.
                - "y" (float): The y-coordinate in kilometers.
                - "z" (float): The z-coordinate in kilometers.
                - "frame" (str, optional): The reference-frame,
                                           see :class:`eosimutil.base.ReferenceFrame`.

        Returns:
            Cartesian3DPosition: Cartesian3DPosition object.
        """
        frame = (
            ReferenceFrame.get(dict_in["frame"]) if "frame" in dict_in else None
        )
        return cls(dict_in["x"], dict_in["y"], dict_in["z"], frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Cartesian3DPosition object to a dictionary.

        Returns:
            dict: Dictionary with the position information.
        """
        return {
            "x": self.coords[0],
            "y": self.coords[1],
            "z": self.coords[2],
            "frame": self.frame.to_string() if self.frame else None,
        }


class Cartesian3DVelocity:
    """Handles 3D velocity information. 
       Internally the velocity is stored in kilometers-per-second.
    """

    def __init__(
        self, vx: float, vy: float, vz: float, frame: Optional[ReferenceFrame]
    ) -> None:
        if not all(isinstance(coord, (int, float)) for coord in [vx, vy, vz]):
            raise ValueError("vx, vy, and vz must be numeric values.")
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError("frame must be a ReferenceFrame object or None.")
        self.coords = np.array([vx, vy, vz])
        self.frame = frame

    @staticmethod
    def from_array(
        array_in: Union[List[float], np.ndarray, Tuple[float, float, float]],
        frame: Optional[Union[ReferenceFrame, str, None]] = None,
    ) -> "Cartesian3DVelocity":
        """Construct a Cartesian3DVelocity object from a list, tuple, or NumPy array.

        Args:
            array_in (Union[List[float], np.ndarray, Tuple[float, float, float]]):
                Velocity coordinates in kilometers-per-second.
            frame (Union[ReferenceFrame, str, None]): Reference-frame.

        Returns:
            Cartesian3DVelocity: Cartesian3DVelocity object.
        """
        if isinstance(array_in, np.ndarray):
            array_in = array_in.tolist()  # Convert NumPy array to list
        elif isinstance(array_in, tuple):
            array_in = list(array_in)  # Convert tuple to list

        if len(array_in) != 3:
            raise ValueError("The input must contain exactly 3 elements.")
        if not all(isinstance(coord, (int, float)) for coord in array_in):
            raise ValueError(
                "All elements in the input must be numeric values."
            )
        if isinstance(frame, str):
            frame = ReferenceFrame.get(frame)
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError(
                "frame must be a ReferenceFrame object, a valid string, or None."
            )
        return Cartesian3DVelocity(array_in[0], array_in[1], array_in[2], frame)

    def to_numpy(self) -> np.ndarray:
        """Convert the Cartesian3DVelocity object to a NumPy array.

        Returns:
            np.ndarray: Velocity coordinates in kilometers-per-second.
        """
        return self.coords

    def to_list(self) -> List[float]:
        """Convert the Cartesian3DVelocity object to a list.

        Returns:
            List[float]: Velocity coordinates in kilometers-per-second.
        """
        return self.coords.tolist()

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "Cartesian3DVelocity":
        """Construct a Cartesian3DVelocity object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the velocity information.
                The dictionary should contain the following key-value
                pairs:
                - "vx" (float): The x-coordinate in km-per-s.
                - "vy" (float): The y-coordinate in km-per-s.
                - "vz" (float): The z-coordinate in km-per-s.
                - "frame" (str, optional): The reference-frame,
                                           see :class:`eosimutil.base.ReferenceFrame`.

        Returns:
            Cartesian3DVelocity: Cartesian3DVelocity object.
        """
        frame = (
            ReferenceFrame.get(dict_in["frame"]) if "frame" in dict_in else None
        )
        return cls(dict_in["vx"], dict_in["vy"], dict_in["vz"], frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Cartesian3DVelocity object to a dictionary.

        Returns:
            dict: Dictionary with the velocity information.
        """
        return {
            "vx": self.coords[0],
            "vy": self.coords[1],
            "vz": self.coords[2],
            "frame": self.frame.to_string() if self.frame else None,
        }


class GeographicPosition:
    """Handles geographic position in the geodetic coordinate system.
    The geodetic position is defined with respect to the
    World Geodetic System 1984 Geoid as defined in Skyfield.
    Reference: https://rhodesmill.org/skyfield/api-topos.html
    """

    def __init__(
        self,
        latitude_degrees: float,
        longitude_degrees: float,
        elevation_m: float,
    ):
        """
        Args:
            latitude_degrees (float): WGS84 Geodetic latitude in degrees.
            longitude_degrees (float): WGS84 Geodetic longitude in degrees.
            elevation_m (float): Elevation in meters.
        """

        self.latitude_degrees = latitude_degrees
        self.longitude_degrees = longitude_degrees
        self.elevation_m = elevation_m

        self.skyfield_geo_position = skyfield_wgs84.latlon(
            latitude_degrees=latitude_degrees,
            longitude_degrees=longitude_degrees,
            elevation_m=elevation_m,
        )

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "GeographicPosition":
        """Construct a GeographicPosition object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the geographic position information.
                The dictionary should contain the following key-value pairs:
                - "latitude" (float): Latitude in degrees.
                - "longitude" (float): Longitude in degrees.
                - "elevation" (float): Elevation in meters.

        Returns:
            GeographicPosition: GeographicPosition object.
        """
        latitude_degrees = dict_in["latitude"]
        longitude_degrees = dict_in["longitude"]
        elevation_m = dict_in["elevation"]
        return cls(latitude_degrees, longitude_degrees, elevation_m)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the GeographicPosition object to a dictionary.

        Returns:
            dict: Dictionary with the geographic position information.
        """
        return {
            "latitude": self.latitude_degrees,
            "longitude": self.longitude_degrees,
            "elevation": self.elevation_m,
        }

    @property
    def latitude(self):
        """Get the latitude in degrees."""
        return self.latitude_degrees

    @property
    def longitude(self):
        """Get the longitude in degrees."""
        return self.longitude_degrees

    @property
    def elevation(self):
        """Get the elevation in meters."""
        return self.elevation_m

    @property
    def itrs_xyz(self):
        """Get the ITRS XYZ position in kilometers.
        Conversion is performed using Skyfield.
        Returns:
            np.ndarray: ITRS XYZ position in kilometers.
        """
        skyfield_geo_position = skyfield_wgs84.latlon(
            latitude_degrees=self.latitude_degrees,
            longitude_degrees=self.longitude_degrees,
            elevation_m=self.elevation_m,
        )
        itrs_xyz = skyfield_geo_position.itrs_xyz.km
        return itrs_xyz


class CartesianState:
    """Handles Cartesian state information."""

    def __init__(
        self,
        time: AbsoluteDate,
        position: Cartesian3DPosition,
        velocity: Cartesian3DVelocity,
        frame: Optional[ReferenceFrame],
    ) -> None:
        if frame is None:
            frame = (
                position.frame if position.frame is not None else velocity.frame
            )
        if position.frame is not None and position.frame != frame:
            raise ValueError(
                "Position frame does not match the provided frame."
            )
        if velocity.frame is not None and velocity.frame != frame:
            raise ValueError(
                "Velocity frame does not match the provided frame."
            )
        self.time: AbsoluteDate = time
        self.position: Cartesian3DPosition = position
        self.velocity: Cartesian3DVelocity = velocity
        self.frame: ReferenceFrame = frame

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "CartesianState":
        """Construct a CartesianState object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the Cartesian state information.
                The dictionary should contain the following key-value pairs:
                - "time" (dict): Dictionary with the date-time information.
                        See  :class:`orbitpy.util.AbsoluteDate.from_dict()`.
                - "frame" (str, optional): The reference-frame
                                           See :class:`eosimutil.base.ReferenceFrame`.
                - "position" (List[float]): Position vector in kilometers.
                - "velocity" (List[float]): Velocity vector in km-per-s.

        Returns:
            CartesianState: CartesianState object.
        """
        time = AbsoluteDate.from_dict(dict_in["time"])
        frame = (
            ReferenceFrame.get(dict_in["frame"]) if "frame" in dict_in else None
        )
        position = Cartesian3DPosition.from_array(dict_in["position"], frame)
        velocity = Cartesian3DVelocity.from_array(dict_in["velocity"], frame)
        return cls(time, position, velocity, frame)

    @staticmethod
    def from_array(
        array_in: Union[
            List[float],
            np.ndarray,
            Tuple[float, float, float, float, float, float],
        ],
        time: AbsoluteDate,
        frame: Optional[Union[ReferenceFrame, str, None]] = None,
    ) -> "CartesianState":
        """Construct a CartesianState object from a list, tuple, or NumPy array.

        Args:
            array_in (Union[List[float], np.ndarray, Tuple[float, float, float, float, float, float]]): # pylint: disable=line-too-long
                Position and velocity coordinates in kilometers and km-per-s.
            time (AbsoluteDate): Absolute date-time object.
            frame (Union[ReferenceFrame, str, None]): Reference-frame.
                If None, the frame will be taken from the position and velocity
                objects.
        Returns:
            CartesianState: CartesianState object.
        """
        if isinstance(array_in, np.ndarray):
            array_in = array_in.tolist()
        elif isinstance(array_in, tuple):
            array_in = list(array_in)
        if len(array_in) != 6:
            raise ValueError("The input must contain exactly 6 elements.")
        if not all(isinstance(coord, (int, float)) for coord in array_in):
            raise ValueError(
                "All elements in the input must be numeric values."
            )
        if isinstance(frame, str):
            frame = ReferenceFrame.get(frame)
        if frame is not None and not isinstance(frame, ReferenceFrame):
            raise ValueError(
                "frame must be a ReferenceFrame object, a valid string, or None."
            )
        position = Cartesian3DPosition.from_array(array_in[0:3], frame)
        velocity = Cartesian3DVelocity.from_array(array_in[3:6], frame)
        return CartesianState(time, position, velocity, frame)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CartesianState object to a dictionary.

        Returns:
            dict: Dictionary with the Cartesian state information.
        """
        return {
            "time": self.time.to_dict(),
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
            "frame": self.frame.to_string(),
        }

    def to_numpy(self) -> np.ndarray:
        """Output the position and velocity in a single NumPy array.

        The resulting array will have a length of 6, containing the
        position (x, y, z) and velocity (vx, vy, vz) components.

        Returns:
            np.ndarray: NumPy array with the position and velocity information.
        """
        return np.concatenate(
            (self.position.to_numpy(), self.velocity.to_numpy())
        )

    def to_skyfield_gcrf_position(self):
        """Convert the CartesianState object to a Skyfield position object.
        The Skyfield "position" object contains the position, velocity, time
        information, and is referenced in GCRF.

        Returns:
            Skyfield position (state) object.

        Raises:
            ValueError: If the frame is not ICRF_EC.
        """
        if self.frame != ReferenceFrame.get("ICRF_EC"):
            raise ValueError(
                "Only CartesianState object in ICRF_EC frame is supported for "
                "conversion to Skyfield GCRF position."
            )

        skyfield_time = self.time.to_skyfield_time()
        position_au = self.position.coords / Skyfield_AU_KM  # convert to AU
        velocity_au_per_d = (
            self.velocity.coords / Skyfield_AU_KM * 86400.0
        )  # convert to AU/day
        return skyfield_build_position(
            position_au=position_au,
            velocity_au_per_d=velocity_au_per_d,
            t=skyfield_time,
            center=399,  # Geocentric
            target=None,
        )

class Cartesian3DPositionArray:
    """Stores an array of 3D Cartesian positions in a specified reference frame.

    The instance can be initialized from a list of Cartesian3DPosition objects
    or from a list of GeographicPosition objects. In the GeographicPosition case,
    latitude, longitude, and elevation are converted to Cartesian coordinates in the
    ITRF frame using the `itrs_xyz` property.

    Internally, the positions are stored as a NumPy array.

    +---------------------------------------------+
    |         Cartesian3DPositionArray            |
    +---------------------------------------------+
    |                                             |
    | positions:                                  |
    | +-----------------------------------------+ |
    | | [[x1, y1, z1],                          | |
    | |  [x2, y2, z2],                          | |
    | |   ...                                   | |  --> NumPy array of shape (N, 3)
    | |  [xN, yN, zN]]                          | |      Units are in kilometers.
    | +-----------------------------------------+ |
    |                                             |
    | frame:                                      |
    | +-----------------------------------------+ |
    | | ReferenceFrame (e.g., ITRF, ICRF_EC)    | |  --> Reference frame for all positions
    | +-----------------------------------------+ |
    +---------------------------------------------+
    """

    def __init__(self, positions: np.ndarray, frame: ReferenceFrame) -> None:
        """Initializes a Cartesian3DPositionArray from a NumPy array and a reference frame.

        Args:
            positions (np.ndarray): NumPy array of shape (N, 3) containing x, y, z points.
            frame (ReferenceFrame): The reference frame for the positions.

        Raises:
            ValueError: If positions does not have shape (N, 3).
        """
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions array must have shape (N, 3)")
        self.positions = positions
        self.frame = frame

    @classmethod
    def from_cartesian_positions(cls, positions: list) -> "Cartesian3DPositionArray":
        """Creates a Cartesian3DPositionArray from a list of Cartesian3DPosition objects.

        Args:
            positions (list): List of Cartesian3DPosition objects.

        Returns:
            Cartesian3DPositionArray: A new Cartesian3DPositionArray object.

        Raises:
            ValueError: If the list is empty or if the frames of the positions do not match.
        """
        if not positions:
            raise ValueError("The list of Cartesian3DPosition objects cannot be empty.")
        frame = positions[0].frame
        if any(pos.frame != frame for pos in positions):
            raise ValueError("All Cartesian3DPosition objects must have the same reference frame.")
        coords = np.array([pos.to_list() for pos in positions])
        return cls(coords, frame)

    @classmethod
    def from_geographic_positions(cls, positions: list) -> "Cartesian3DPositionArray":
        """Creates a Cartesian3DPositionArray from a list of GeographicPosition objects.

        Geographic positions are converted into Cartesian coordinates in the ITRF frame
        using the `itrs_xyz` property.

        Args:
            positions (list): List of GeographicPosition objects.

        Returns:
            Cartesian3DPositionArray: A new Cartesian3DPositionArray object.

        Raises:
            ValueError: If the list is empty.
        """
        if not positions:
            raise ValueError("The list of GeographicPosition objects cannot be empty.")
        frame = ReferenceFrame.get("ITRF")
        coords = np.array([pos.itrs_xyz for pos in positions])
        return cls(coords, frame)

    def to_numpy(self) -> np.ndarray:
        """Returns the internal NumPy array of positions.

        Returns:
            np.ndarray: Array of shape (N, 3) containing the positions.
        """
        return self.positions

    def to_list(self) -> list:
        """Returns a list of positions.

        Returns:
            list: List of [x, y, z] coordinates.
        """
        return self.positions.tolist()

    @classmethod
    def from_dict(cls, dict_in: dict) -> "Cartesian3DPositionArray":
        """Deserializes a Cartesian3DPositionArray object from a dictionary.

        Args:
            dict_in (dict): The dictionary must contain the following keys:
                - "positions": A list or array representing positions (Nx3).
                - "frame": A string representing the reference frame.

        Returns:
            Cartesian3DPositionArray: The deserialized object.
        """
        positions = np.array(dict_in["positions"])
        frame = ReferenceFrame.get(dict_in["frame"]) if "frame" in dict_in and dict_in["frame"] else None
        return cls(positions, frame)

    def to_dict(self) -> dict:
        """Serializes the Cartesian3DPositionArray to a dictionary.

        Returns:
            dict: A dictionary representation of the Cartesian3DPositionArray object.
        """
        return {
            "positions": self.to_list(),
            "frame": self.frame.to_string() if self.frame is not None else None,
        }

    def __repr__(self) -> str:
        """Returns the string representation of the Cartesian3DPositionArray object."""
        return f"Cartesian3DPositionArray(positions={self.positions!r}, frame={self.frame!r})"
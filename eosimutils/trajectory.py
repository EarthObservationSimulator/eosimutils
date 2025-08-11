"""
.. module:: eosimutils.trajectory
   :synopsis: Trajectory data representation.

This module provides a StateSeries class that stores a vector of times and separate numpy arrays
for position (km) and velocity (km/s), and a PositionSeries class that stores a vector of times
and a single numpy array for position (km). The time is represented as an AbsoluteDateArray object.
Missing data (gaps) are represented by NaN values.

Basic interpolation/resampling and arithmetic operations (with frame conversion) are supported.
"""

# pylint: disable=protected-access

import numpy as np
import spiceypy as spice
from typing import Union

from .base import ReferenceFrame
from .time import AbsoluteDate, AbsoluteDateArray
from .timeseries import Timeseries
from .spicekernels import load_spice_kernels


def convert_frame(
    positions: np.ndarray,
    velocities: np.ndarray,
    times: np.ndarray,
    from_frame: ReferenceFrame,
    to_frame: ReferenceFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts position and velocity arrays from one reference frame to another using SPICE.

    Args:
        positions (np.ndarray): Array of shape (N, 3) representing positions in `from_frame`.
        velocities (np.ndarray): Array of shape (N, 3) representing velocities in `from_frame`.
        times (np.ndarray): Array of time samples corresponding to each state in ephemeris time.
        from_frame (ReferenceFrame): The original reference frame.
        to_frame (ReferenceFrame): The target reference frame.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - positions (np.ndarray): Converted positions in `to_frame`.
            - velocities (np.ndarray): Converted velocities in `to_frame`.

    Raises:
        NotImplementedError: If the frame conversion is not implemented.
    """
    if from_frame == to_frame:
        return positions, velocities

    if from_frame.to_string() == "ITRF":
        from_string = "ITRF93"
    elif from_frame.to_string() == "ICRF_EC":
        from_string = "J2000"
    else:
        raise NotImplementedError(
            f"Frame conversion from {from_frame} to {to_frame} is not implemented."
        )

    if to_frame.to_string() == "ITRF":
        to_string = "ITRF93"
    elif to_frame.to_string() == "ICRF_EC":
        to_string = "J2000"
    else:
        raise NotImplementedError(
            f"Conversion from {from_frame} to {to_frame} is not implemented."
        )

    new_positions = np.empty_like(positions)
    new_velocities = np.empty_like(velocities)
    for i, t in enumerate(times):
        t_matrix = spice.sxform(from_string, to_string, t)
        new_positions[i] = t_matrix[0:3, 0:3].dot(positions[i]) + t_matrix[
            0:3, 3:6
        ].dot(velocities[i])
        new_velocities[i] = t_matrix[3:6, 0:3].dot(positions[i]) + t_matrix[
            3:6, 3:6
        ].dot(velocities[i])
    return new_positions, new_velocities


def convert_frame_position(
    positions: np.ndarray,
    times: np.ndarray,
    from_frame: ReferenceFrame,
    to_frame: ReferenceFrame,
) -> np.ndarray:
    """
    Converts position arrays from one reference frame to another using SPICE.

    Args:
        positions (np.ndarray): Array of shape (N, 3) representing positions in `from_frame`.
        times (np.ndarray): Array of time samples corresponding to each position in ephemeris time.
        from_frame (ReferenceFrame): The original reference frame.
        to_frame (ReferenceFrame): The target reference frame.

    Returns:
        np.ndarray: Converted positions in `to_frame`.

    Raises:
        NotImplementedError: If the frame conversion is not implemented.
    """
    if from_frame == to_frame:
        return positions

    if from_frame.to_string() == "ITRF":
        from_string = "J2000"
    elif from_frame.to_string() == "ICRF_EC":
        from_string = "ITRF93"
    else:
        raise NotImplementedError(
            f"Frame conversion from {from_frame} to {to_frame} is not implemented."
        )

    if to_frame.to_string() == "ITRF":
        to_string = "J2000"
    elif to_frame.to_string() == "ICRF_EC":
        to_string = "ITRF93"
    else:
        raise NotImplementedError(
            f"Conversion from {from_frame} to {to_frame} is not implemented."
        )

    new_positions = np.empty_like(positions)
    for i, t in enumerate(times):
        t_matrix = spice.pxform(from_string, to_string, t)
        new_positions[i] = t_matrix.dot(positions[i])
    return new_positions


class StateSeries(Timeseries):
    """
    Represents trajectory data as a timeseries with separate arrays for position and velocity.

    +-----------------------------------+
    |           StateSeries             |
    +-----------------------------------+
    |                                   |
    | time:                             |
    | +-------------------------------+ |
    | | t1, t2, t3, ..., tN           | |  --> AbsoluteDateArray (ephemeris times)
    | +-------------------------------+ |
    |                                   |
    | data:                             |
    | +-------------------------------+ |
    | | Positions (Nx3):              | |
    | | [[x1, y1, z1],                | |
    | |  [x2, y2, z2],                | |  --> Position array in kilometers
    | |  ...,                         | |
    | |  [xN, yN, zN]]                | |
    | +-------------------------------+ |
    | | Velocities (Nx3):             | |
    | | [[vx1, vy1, vz1],             | |
    | |  [vx2, vy2, vz2],             | |  --> Velocity array in kilometers per second
    | |  ...,                         | |
    | |  [vxN, vyN, vzN]]             | |
    | +-------------------------------+ |
    |                                   |
    | headers:                          |
    | +-------------------------------+ |
    | | ["pos_x", "pos_y", "pos_z"],  | |  --> Labels for position components
    | | ["vel_x", "vel_y", "vel_z"]   | |  --> Labels for velocity components
    | +-------------------------------+ |
    |                                   |
    | frame:                            |
    | +-------------------------------+ |
    | | ReferenceFrame (e.g., ITRF)   | |  --> Reference frame for the trajectory
    | +-------------------------------+ |
    +-----------------------------------+

    Attributes:
        time (AbsoluteDateArray): A vector of time samples.
        data (list): A list containing position (Nx3) and velocity (Nx3) arrays.
        headers (list): Nested labels for position and velocity.
        frame (ReferenceFrame): The reference frame for the trajectory.
    """

    def __init__(
        self, time: "AbsoluteDateArray", data: list, frame: ReferenceFrame
    ):
        """
        Initializes a StateSeries object.

        Args:
            time (AbsoluteDateArray): A vector of time samples.
            data (list): A list containing two arrays:
                - Position array of shape (N, 3).
                - Velocity array of shape (N, 3).
            frame (ReferenceFrame): The reference frame for the trajectory.

        Raises:
            ValueError: If `data` does not contain two arrays of shape (N, 3).
        """
        if len(data) != 2 or data[0].shape[1] != 3 or data[1].shape[1] != 3:
            raise ValueError(
                "Data must be a list containing two arrays: pos (Nx3) and vel (Nx3)."
            )
        headers = [["pos_x", "pos_y", "pos_z"], ["vel_x", "vel_y", "vel_z"]]
        super().__init__(time, data, headers=headers)
        self.frame = frame
        load_spice_kernels()  # Load SPICE kernels for frame conversion

    def resample(
        self, new_time: AbsoluteDateArray, method: str = "linear"
    ) -> "StateSeries":
        """
        Resamples the StateSeries to a new time base.

        Takes AbsoluteDataArray as input and calls the private _resample method.

        Args:
            new_time (AbsoluteDateArray): The new time samples.
            method (str, optional): Interpolation method. Defaults to "linear".

        Returns:
            StateSeries: A new StateSeries object with resampled data.
        """
        return self._resample(new_time.ephemeris_time, method)

    def _resample(
        self, new_time: np.ndarray, method: str = "linear"
    ) -> "StateSeries":
        """
        Resamples the StateSeries to a new time base.

        This method works by interpolating the data arrays to the new time samples.
        It considers the position and velocity seperately (i.e., velocity is not used
        to help interpolate posiiton).

        Args:
            new_time (np.ndarray): The new time samples in ephemeris time.
            method (str): The interpolation method to use. Defaults to "linear".

        Returns:
            StateSeries: A new StateSeries object with resampled data.
        """
        new_time_obj, new_data, _ = self._resample_data(new_time, method)
        return StateSeries(new_time_obj, new_data, self.frame)

    def remove_gaps(self) -> "StateSeries":
        """
        Removes gaps (NaN values) from the StateSeries.

        Returns:
            StateSeries: A new StateSeries object with gaps removed.
        """
        new_time, new_data, _ = self._remove_gaps_data()
        return StateSeries(new_time, new_data, self.frame)

    def _arithmetic_op(self, other, op, interp_method: str = "linear"):
        """
        Perform arithmetic operations (e.g., addition, subtraction) between trajectories
        or with a scalar.

        This method ensures that reference frames are compatible before performing the operation.

        Args:
            other (StateSeries or scalar): The operand for the operation.
            op (callable): The operation to perform (e.g., addition, subtraction).
            interp_method (str): The interpolation method to use. Defaults to "linear".

        Returns:
            StateSeries: A new StateSeries object with the result of the operation.

        Raises:
            TypeError: If the operand is neither a StateSeries nor a scalar.
        """
        if np.isscalar(other):
            # Delegate scalar operations to the parent class.
            return super()._arithmetic_op(other, op)
        elif isinstance(other, StateSeries):
            # Resample other onto self.time.ephemeris_time (using the underlying ephemeris times).
            other_resamp = other._resample(
                self.time.ephemeris_time, method=interp_method
            )
            # If frames do not match, attempt frame conversion.
            if self.frame != other.frame:
                pos_conv, vel_conv = convert_frame(
                    other_resamp.data[0],
                    other_resamp.data[1],
                    self.time.ephemeris_time,
                    other.frame,
                    self.frame,
                )
                other_resamp_data = [pos_conv, vel_conv]
            else:
                other_resamp_data = other_resamp.data
            # Perform vectorized operation for each data array.
            new_data = [
                op(arr, other_arr)
                for arr, other_arr in zip(self.data, other_resamp_data)
            ]
            return StateSeries(self.time, new_data, self.frame)
        else:
            raise TypeError("Operand must be a StateSeries or a scalar.")

    def __add__(self, other):
        """
        Adds another StateSeries or scalar to this StateSeries.

        Args:
            other (StateSeries or scalar): The operand for addition.

        Returns:
            StateSeries: A new StateSeries object with the result of the addition.
        """
        return self._arithmetic_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        """
        Subtracts another StateSeries or scalar from this StateSeries.

        Args:
            other (StateSeries or scalar): The operand for subtraction.

        Returns:
            StateSeries: A new StateSeries object with the result of the subtraction.
        """
        return self._arithmetic_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        """
        Multiplies this StateSeries by another StateSeries or scalar.

        Args:
            other (StateSeries or scalar): The operand for multiplication.

        Returns:
            StateSeries: A new StateSeries object with the result of the multiplication.
        """
        return self._arithmetic_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        """
        Divides this StateSeries by another StateSeries or scalar.

        Args:
            other (StateSeries or scalar): The operand for division.

        Returns:
            StateSeries: A new StateSeries object with the result of the division.
        """
        return self._arithmetic_op(other, lambda a, b: a / b)

    def to_frame(self, to_frame: ReferenceFrame) -> "StateSeries":
        """
        Converts the StateSeries's reference frame to a new frame.

        Args:
            to_frame (ReferenceFrame): The target reference frame.

        Returns:
            StateSeries: A new StateSeries object in the target frame.
        """
        if self.frame == to_frame:
            return self
        pos, vel = convert_frame(
            self.data[0],
            self.data[1],
            self.time.ephemeris_time,
            self.frame,
            to_frame,
        )
        return StateSeries(
            AbsoluteDateArray(self.time.ephemeris_time.copy()),
            [pos, vel],
            to_frame,
        )

    def to_dict(self) -> dict:
        """
        Serializes the StateSeries object to a dictionary.

        Returns:
            dict: A dictionary representation of the StateSeries object.
        """
        return {
            "time": self.time.to_dict(),
            "data": [arr.tolist() for arr in self.data],
            "frame": self.frame.to_string(),
            "headers": self.headers,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> "StateSeries":
        """
        Deserializes a StateSeries object from a dictionary.

        The dictionary must contain the following keys:
            - "time": A dictionary representing the AbsoluteDateArray.
            - "data": A list of two arrays:
                - The first array represents positions (Nx3).
                - The second array represents velocities (Nx3).
            - "frame": A string representing the reference frame.
            - "headers": A nested list of labels for position and velocity.

        Args:
            dct (dict): A dictionary representation of a StateSeries object.

        Returns:
            StateSeries: The deserialized StateSeries object.

        Examples:
            dct = {
                "time": {"et": [0.0, 1.0]},
                "data": [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Positions
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]   # Velocities
                ],
                "frame": "J2000",
                "headers": [["pos_x", "pos_y", "pos_z"], ["vel_x", "vel_y", "vel_z"]]
            }

            trajectory = StateSeries.from_dict(dct)
        """
        time = AbsoluteDateArray.from_dict(dct["time"])
        data_arrays = [np.array(arr) for arr in dct["data"]]
        frame = ReferenceFrame(dct["frame"])
        return cls(time, data_arrays, frame)

    @classmethod
    def constant_position(
        cls, t1: float, t2: float, position: np.ndarray, frame: ReferenceFrame
    ) -> "StateSeries":
        """
        Creates a constant position trajectory with zero velocity.

        Args:
            t1 (float): The start time.
            t2 (float): The end time.
            position (np.ndarray): A 3-element array representing the position.
            frame (ReferenceFrame): The reference frame for the trajectory.

        Returns:
            StateSeries: A new StateSeries object with constant position.

        Raises:
            ValueError: If `position` is not a 3-element array.
        """
        if position.shape != (3,):
            raise ValueError("position must be a 3-element array.")
        time_obj = AbsoluteDateArray(np.array([t1, t2]))
        traj = np.column_stack((np.tile(position, (2, 1)), np.zeros((2, 3))))
        return cls(time_obj, [traj[:, :3], traj[:, 3:]], frame)

    @classmethod
    def constant_velocity(
        cls,
        t1: float,
        t2: float,
        velocity: np.ndarray,
        initial_position: np.ndarray,
        frame: ReferenceFrame,
    ) -> "StateSeries":
        """
        Creates a constant velocity trajectory.

        Args:
            t1 (float): The start time.
            t2 (float): The end time.
            velocity (np.ndarray): A 3-element array representing the velocity.
            initial_position (np.ndarray): A 3-element array representing the initial position.
            frame (ReferenceFrame): The reference frame for the trajectory.

        Returns:
            StateSeries: A new StateSeries object with constant velocity.

        Raises:
            ValueError: If `velocity` or `initial_position` is not a 3-element array.
        """
        if velocity.shape != (3,) or initial_position.shape != (3,):
            raise ValueError(
                "velocity and initial_position must be 3-element arrays."
            )
        dt = t2 - t1
        final_position = initial_position + velocity * dt
        time_obj = AbsoluteDateArray(np.array([t1, t2]))
        traj = np.column_stack(
            (
                np.vstack((initial_position, final_position)),
                np.tile(velocity, (2, 1)),
            )
        )
        return cls(time_obj, [traj[:, :3], traj[:, 3:]], frame)

    @classmethod
    def from_list_of_cartesian_state(cls, states: list) -> "StateSeries":
        """
        Creates a StateSeries object from a list of CartesianState objects.

        Args:
            states (list): A list of CartesianState objects.

        Returns:
            StateSeries: A new StateSeries object.

        Raises:
            ValueError: If the list is empty or if the frames of the CartesianState objects do not
            match.
        """
        if not states:
            raise ValueError(
                "The list of CartesianState objects cannot be empty."
            )

        # Extract the frame from the first state and ensure all frames match
        frame = states[0].frame
        if any(state.frame != frame for state in states):
            raise ValueError(
                "All CartesianState objects must have the same reference frame."
            )

        # Extract time, position, and velocity data
        times = np.array([state.time.ephemeris_time for state in states])
        positions = np.array([state.position.to_numpy() for state in states])
        velocities = np.array([state.velocity.to_numpy() for state in states])

        # Create an AbsoluteDateArray for the time
        time_obj = AbsoluteDateArray(times)

        # Return a new StateSeries object
        return cls(time_obj, [positions, velocities], frame)


class PositionSeries(Timeseries):
    """
    Represents position data as a timeseries with a single array for position.

    +-----------------------------------+
    |           PositionSeries          |
    +-----------------------------------+
    |                                   |
    | time:                             |
    | +-------------------------------+ |
    | | t1, t2, t3, ..., tN           | |  --> AbsoluteDateArray (ephemeris times)
    | +-------------------------------+ |
    |                                   |
    | data:                             |
    | +-------------------------------+ |
    | | Positions (Nx3):              | |
    | | [[x1, y1, z1],                | |
    | |  [x2, y2, z2],                | |  --> Position array in kilometers
    | |  ...,                         | |
    | |  [xN, yN, zN]]                | |
    | +-------------------------------+ |
    |                                   |
    | headers:                          |
    | +-------------------------------+ |
    | | ["pos_x", "pos_y", "pos_z"]   | |  --> Labels for position components
    | +-------------------------------+ |
    |                                   |
    | frame:                            |
    | +-------------------------------+ |
    | | ReferenceFrame (e.g., ITRF)   | |  --> Reference frame for the position data
    | +-------------------------------+ |
    +-----------------------------------+

    Attributes:
        time (AbsoluteDateArray): A vector of time samples.
        data (np.ndarray): A single array containing position (Nx3).
        headers (list): Labels for position components.
        frame (ReferenceFrame): The reference frame for the position data.
    """

    def __init__(
        self, time: "AbsoluteDateArray", data: np.ndarray, frame: ReferenceFrame
    ):
        """
        Initializes a PositionSeries object.

        Args:
            time (AbsoluteDateArray): A vector of time samples.
            data (np.ndarray): An array of shape (N, 3) representing positions.
            frame (ReferenceFrame): The reference frame for the position data.

        Raises:
            ValueError: If `data` does not have shape (N, 3).
        """
        if data.shape[1] != 3:
            raise ValueError(
                "Data must be an array of shape (N, 3) for positions."
            )
        headers = ["pos_x", "pos_y", "pos_z"]
        super().__init__(time, [data], headers=[headers])
        self.frame = frame
        load_spice_kernels()  # Load SPICE kernels for frame conversion

    def resample(
        self, new_time: np.ndarray, method: str = "linear"
    ) -> "PositionSeries":
        """
        Resamples the position data to a new time base.

        Args:
            new_time (np.ndarray): The new time samples.
            method (str): The interpolation method to use. Defaults to "linear".

        Returns:
            PositionSeries: A new PositionSeries object with resampled data.
        """
        new_time_obj, new_data, _ = self._resample_data(new_time, method)
        return PositionSeries(new_time_obj, new_data[0], self.frame)

    def remove_gaps(self) -> "PositionSeries":
        """
        Removes gaps (NaN values) from the position data.

        Returns:
            PositionSeries: A new PositionSeries object with gaps removed.
        """
        new_time, new_data, _ = self._remove_gaps_data()
        return PositionSeries(new_time, new_data[0], self.frame)

    def to_frame(self, to_frame: ReferenceFrame) -> "PositionSeries":
        """
        Converts the position data's reference frame to a new frame.

        Args:
            to_frame (ReferenceFrame): The target reference frame.

        Returns:
            PositionSeries: A new PositionSeries object in the target frame.
        """
        if self.frame == to_frame:
            return self
        pos = convert_frame_position(
            self.data[0],
            self.time.ephemeris_time,
            self.frame,
            to_frame,
        )
        return PositionSeries(
            AbsoluteDateArray(self.time.ephemeris_time.copy()), pos, to_frame
        )

    def to_dict(self) -> dict:
        """
        Serializes the PositionSeries object to a dictionary.

        Returns:
            dict: A dictionary representation of the PositionSeries object.
        """
        return {
            "time": self.time.to_dict(),
            "data": self.data[0].tolist(),
            "frame": self.frame.to_string(),
            "headers": self.headers[0],
        }

    @classmethod
    def from_dict(cls, dct: dict) -> "PositionSeries":
        """
        Deserializes a PositionSeries object from a dictionary.

        Args:
            dct (dict): A dictionary representation of a PositionSeries object.

        Returns:
            PositionSeries: The deserialized PositionSeries object.
        """
        time = AbsoluteDateArray.from_dict(dct["time"])
        data_array = np.array(dct["data"])
        frame = ReferenceFrame(dct["frame"])
        return cls(time, data_array, frame)

    @classmethod
    def constant_position(
        cls, t1: float, t2: float, position: np.ndarray, frame: ReferenceFrame
    ) -> "PositionSeries":
        """
        Creates a constant position series.

        Args:
            t1 (float): The start time.
            t2 (float): The end time.
            position (np.ndarray): A 3-element array representing the position.
            frame (ReferenceFrame): The reference frame for the position data.

        Returns:
            PositionSeries: A new PositionSeries object with constant position.

        Raises:
            ValueError: If `position` is not a 3-element array.
        """
        if position.shape != (3,):
            raise ValueError("position must be a 3-element array.")
        time_obj = AbsoluteDateArray(np.array([t1, t2]))
        pos_data = np.tile(position, (2, 1))
        return cls(time_obj, pos_data, frame)

    @classmethod
    def from_list_of_cartesian_position(
        cls, positions: list
    ) -> "PositionSeries":
        """
        Creates a PositionSeries object from a list of Cartesian3DPosition objects.

        Args:
            positions (list): A list of Cartesian3DPosition objects.

        Returns:
            PositionSeries: A new PositionSeries object.

        Raises:
            ValueError: If the list is empty or if the frames of the Cartesian3DPosition objects
            do not match.
        """
        if not positions:
            raise ValueError(
                "The list of Cartesian3DPosition objects cannot be empty."
            )

        # Extract the frame from the first position and ensure all frames match
        frame = positions[0].frame
        if any(pos.frame != frame for pos in positions):
            raise ValueError(
                "All Cartesian3DPosition objects must have the same reference frame."
            )

        # Extract time and position data
        times = np.array([pos.time.ephemeris_time for pos in positions])
        pos_data = np.array([pos.to_numpy() for pos in positions])

        # Create an AbsoluteDateArray for the time
        time_obj = AbsoluteDateArray(times)

        # Return a new PositionSeries object
        return cls(time_obj, pos_data, frame)

    def _arithmetic_op(self, other, op, interp_method: str = "linear"):
        """
        Perform arithmetic operations (e.g., addition, subtraction) between position series
        or with a scalar.

        Args:
            other (PositionSeries or scalar): The operand for the operation.
            op (callable): The operation to perform (e.g., addition, subtraction).
            interp_method (str): The interpolation method to use. Defaults to "linear".

        Returns:
            PositionSeries: A new PositionSeries object with the result of the operation.

        Raises:
            TypeError: If the operand is neither a PositionSeries nor a scalar.
        """
        if np.isscalar(other):
            # Delegate scalar operations to the parent class.
            return super()._arithmetic_op(other, op)
        elif isinstance(other, PositionSeries):
            # Resample other onto self.time.ephemeris_time (using the underlying ephemeris times).
            other_resamp = other.resample(
                self.time.ephemeris_time, method=interp_method
            )
            # If frames do not match, attempt frame conversion.
            if self.frame != other.frame:
                pos_conv = convert_frame_position(
                    other_resamp.data[0],
                    self.time.ephemeris_time,
                    other.frame,
                    self.frame,
                )
                other_resamp_data = pos_conv
            else:
                other_resamp_data = other_resamp.data[0]
            # Perform vectorized operation for the data array.
            new_data = [op(self.data[0], other_resamp_data)]
            return PositionSeries(self.time, new_data[0], self.frame)
        else:
            raise TypeError("Operand must be a PositionSeries or a scalar.")

    def __add__(self, other):
        """
        Adds another PositionSeries or scalar to this PositionSeries.

        Args:
            other (PositionSeries or scalar): The operand for addition.

        Returns:
            PositionSeries: A new PositionSeries object with the result of the addition.
        """
        return self._arithmetic_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        """
        Subtracts another PositionSeries or scalar from this PositionSeries.

        Args:
            other (PositionSeries or scalar): The operand for subtraction.

        Returns:
            PositionSeries: A new PositionSeries object with the result of the subtraction.
        """
        return self._arithmetic_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        """
        Multiplies this PositionSeries by another PositionSeries or scalar.

        Args:
            other (PositionSeries or scalar): The operand for multiplication.

        Returns:
            PositionSeries: A new PositionSeries object with the result of the multiplication.
        """
        return self._arithmetic_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        """
        Divides this PositionSeries by another PositionSeries or scalar.

        Args:
            other (PositionSeries or scalar): The operand for division.

        Returns:
            PositionSeries: A new PositionSeries object with the result of the division.
        """
        return self._arithmetic_op(other, lambda a, b: a / b)

    def at(self, t: Union["AbsoluteDate", "AbsoluteDateArray"]) -> np.ndarray:
        """
        Returns the position(s) at the given time(s) by interpolation.

        Args:
            t (AbsoluteDate or AbsoluteDateArray): The time(s) at which to evaluate the position.

        Returns:
            np.ndarray: Interpolated position(s). Shape (3,) for AbsoluteDate, (N,3) for
            AbsoluteDateArray.
        """
        if isinstance(t, AbsoluteDate):
            query_times = np.array([t.ephemeris_time])
            single = True
        elif isinstance(t, AbsoluteDateArray):
            query_times = t.ephemeris_time
            single = False
        else:
            raise TypeError("Input must be AbsoluteDate or AbsoluteDateArray.")

        # Use internal _resample_data for interpolation
        _, new_data, _ = self._resample_data(query_times, method="linear")
        positions = new_data[0]
        if single:
            return positions[0]
        return positions

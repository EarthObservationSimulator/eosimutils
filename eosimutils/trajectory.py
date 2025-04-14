"""
.. module:: eosimutils.trajectory
   :synopsis: Trajectory data representation.

This module provides a Trajectory class that stores a vector of times and separate numpy arrays
for position (km) and velocity (km/s). Missing data (gaps) are represented by NaN values.
Basic interpolation/resampling and arithmetic operations (with frame conversion) are supported.
"""

import numpy as np
import spiceypy as spice

from .base import ReferenceFrame  # Assumed to exist
from .time import AbsoluteDateArray
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


class Trajectory(Timeseries):
    """
    Represents trajectory data as a timeseries with separate arrays for position and velocity.

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
        Initializes a Trajectory object.

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
        self, new_time: np.ndarray, method: str = "linear"
    ) -> "Trajectory":
        """
        Resamples the trajectory to a new time base.

        This method works by interpolating the data arrays to the new time samples.
        It considers the position and velocity seperately (i.e., velocity is not used
        to help interpolate posiiton).

        Args:
            new_time (np.ndarray): The new time samples.
            method (str): The interpolation method to use. Defaults to "linear".

        Returns:
            Trajectory: A new Trajectory object with resampled data.
        """
        new_time_obj, new_data, _ = self._resample_data(new_time, method)
        return Trajectory(new_time_obj, new_data, self.frame)

    def remove_gaps(self) -> "Trajectory":
        """
        Removes gaps (NaN values) from the trajectory.

        Returns:
            Trajectory: A new Trajectory object with gaps removed.
        """
        new_time, new_data, _ = self._remove_gaps_data()
        return Trajectory(new_time, new_data, self.frame)

    def _arithmetic_op(self, other, op, interp_method: str = "linear"):
        """
        Perform arithmetic operations (e.g., addition, subtraction) between trajectories
        or with a scalar.

        This method ensures that reference frames are compatible before performing the operation.

        Args:
            other (Trajectory or scalar): The operand for the operation.
            op (callable): The operation to perform (e.g., addition, subtraction).
            interp_method (str): The interpolation method to use. Defaults to "linear".

        Returns:
            Trajectory: A new Trajectory object with the result of the operation.

        Raises:
            TypeError: If the operand is neither a Trajectory nor a scalar.
        """
        if np.isscalar(other):
            # Delegate scalar operations to the parent class.
            return super()._arithmetic_op(other, op)
        elif isinstance(other, Trajectory):
            # Resample other onto self.time.et (using the underlying ephemeris times).
            other_resamp = other.resample(self.time.et, method=interp_method)
            # If frames do not match, attempt frame conversion.
            if self.frame != other.frame:
                pos_conv, vel_conv = convert_frame(
                    other_resamp.data[0],
                    other_resamp.data[1],
                    self.time.et,
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
            return Trajectory(self.time, new_data, self.frame)
        else:
            raise TypeError("Operand must be a Trajectory or a scalar.")

    def __add__(self, other):
        """
        Adds another Trajectory or scalar to this Trajectory.

        Args:
            other (Trajectory or scalar): The operand for addition.

        Returns:
            Trajectory: A new Trajectory object with the result of the addition.
        """
        return self._arithmetic_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        """
        Subtracts another Trajectory or scalar from this Trajectory.

        Args:
            other (Trajectory or scalar): The operand for subtraction.

        Returns:
            Trajectory: A new Trajectory object with the result of the subtraction.
        """
        return self._arithmetic_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        """
        Multiplies this Trajectory by another Trajectory or scalar.

        Args:
            other (Trajectory or scalar): The operand for multiplication.

        Returns:
            Trajectory: A new Trajectory object with the result of the multiplication.
        """
        return self._arithmetic_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        """
        Divides this Trajectory by another Trajectory or scalar.

        Args:
            other (Trajectory or scalar): The operand for division.

        Returns:
            Trajectory: A new Trajectory object with the result of the division.
        """
        return self._arithmetic_op(other, lambda a, b: a / b)

    def to_frame(self, to_frame: ReferenceFrame) -> "Trajectory":
        """
        Converts the trajectory's reference frame to a new frame.

        Args:
            to_frame (ReferenceFrame): The target reference frame.

        Returns:
            Trajectory: A new Trajectory object in the target frame.
        """
        if self.frame == to_frame:
            return self
        pos, vel = convert_frame(
            self.data[0], self.data[1], self.time.et, self.frame, to_frame
        )
        return Trajectory(
            AbsoluteDateArray(self.time.et.copy()), [pos, vel], to_frame
        )

    def to_dict(self) -> dict:
        """
        Serializes the Trajectory object to a dictionary.

        Returns:
            dict: A dictionary representation of the Trajectory object.
        """
        return {
            "time": self.time.to_dict(),
            "data": [arr.tolist() for arr in self.data],
            "frame": self.frame.to_string(),
            "headers": self.headers,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> "Trajectory":
        """
        Deserializes a Trajectory object from a dictionary.

        The dictionary must contain the following keys:
            - "time": A dictionary representing the AbsoluteDateArray.
            - "data": A list of two arrays:
                - The first array represents positions (Nx3).
                - The second array represents velocities (Nx3).
            - "frame": A string representing the reference frame.
            - "headers": A nested list of labels for position and velocity.

        Args:
            dct (dict): A dictionary representation of a Trajectory object.

        Returns:
            Trajectory: The deserialized Trajectory object.

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

            trajectory = Trajectory.from_dict(dct)
        """
        time = AbsoluteDateArray.from_dict(dct["time"])
        data_arrays = [np.array(arr) for arr in dct["data"]]
        frame = ReferenceFrame(dct["frame"])
        return cls(time, data_arrays, frame)

    @classmethod
    def constant_position(
        cls, t1: float, t2: float, position: np.ndarray, frame: ReferenceFrame
    ) -> "Trajectory":
        """
        Creates a constant position trajectory with zero velocity.

        Args:
            t1 (float): The start time.
            t2 (float): The end time.
            position (np.ndarray): A 3-element array representing the position.
            frame (ReferenceFrame): The reference frame for the trajectory.

        Returns:
            Trajectory: A new Trajectory object with constant position.

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
    ) -> "Trajectory":
        """
        Creates a constant velocity trajectory.

        Args:
            t1 (float): The start time.
            t2 (float): The end time.
            velocity (np.ndarray): A 3-element array representing the velocity.
            initial_position (np.ndarray): A 3-element array representing the initial position.
            frame (ReferenceFrame): The reference frame for the trajectory.

        Returns:
            Trajectory: A new Trajectory object with constant velocity.

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
    def from_list_of_cartesian_state(cls, states: list) -> "Trajectory":
        """
        Creates a Trajectory object from a list of CartesianState objects.

        Args:
            states (list): A list of CartesianState objects.

        Returns:
            Trajectory: A new Trajectory object.

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

        # Return a new Trajectory object
        return cls(time_obj, [positions, velocities], frame)

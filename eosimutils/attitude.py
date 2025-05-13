"""Module for handling constant and timeseries attitude data."""

import spiceypy as spice

from typing import Optional, Union, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from .frames import ReferenceFrame
from .time import AbsoluteDate, AbsoluteDateArray
from .trajectory import StateSeries


class Attitude:
    """
    Base class for attitude representations.

    Subclasses must implement `at(t)` to return (Rotation, angular_velocity).

    Provides `transform(state, t)` to apply rotation and angular velocity to 6D state vectors.
    """

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray]
    ) -> tuple[Rotation, np.ndarray]:
        """
        Return (rotation, angular_velocity) at the given time(s).

        Args:
            t: AbsoluteDate or AbsoluteDateArray.
        Returns:
            Tuple[Rotation, np.ndarray]: Rotation and angular velocity vector(s).
        """
        raise NotImplementedError

    def transform(
        self, state: np.ndarray, t: Union[AbsoluteDate, AbsoluteDateArray]
    ) -> np.ndarray:
        """
        Transform a 6D state vector or array of state vectors [pos(3), vel(3)] at given time(s).

        Args:
            state: shape (6,) or (N,6).
            t: AbsoluteDate or AbsoluteDateArray of matching length if state is (N,6).

        Returns:
            Transformed state array of same shape.
        """
        rot, w = self.at(t)
        # single time
        if isinstance(t, AbsoluteDate):
            pos = state[:3]
            vel = state[3:]
            new_pos = rot.apply(pos)
            new_vel = rot.apply(vel) + np.cross(w, new_pos)
            return np.concatenate([new_pos, new_vel])
        # multiple times
        pos = state[:, :3]
        vel = state[:, 3:]
        new_pos = rot.apply(pos)
        new_vel = rot.apply(vel) + np.cross(w, new_pos)
        return np.hstack([new_pos, new_vel])


class ConstantAttitude(Attitude):
    """
    Represents a time-invariant attitude using a single constant rotation.

    Attributes:
        rotation (Rotation): The constant orientation.
        base (ReferenceFrame): The base reference frame.
    """

    def __init__(self, rotation: Rotation, base: ReferenceFrame):
        """
        Initialize a ConstantAttitude.

        Args:
            rotation (Rotation): A single scipy Rotation object.
            base (ReferenceFrame): The base frame of the rotation.
        """
        self.rotation = rotation
        self.base = base

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray]
    ) -> tuple[Rotation, np.ndarray]:
        """
        Evaluate the attitude at the given time(s). Always returns the same rotation.

        Args:
            t (AbsoluteDate or AbsoluteDateArray): Input time(s).

        Returns:
            Tuple[Rotation, np.ndarray]: The constant rotation and a zero angular velocity vector.
        """
        if isinstance(t, AbsoluteDate):
            return self.rotation, np.zeros(3)
        elif isinstance(t, AbsoluteDateArray):
            n = len(t.ephemeris_time)
            return Rotation.from_quat([self.rotation.as_quat()] * n), np.zeros(
                (n, 3)
            )
        else:
            raise TypeError("t must be AbsoluteDate or AbsoluteDateArray")

    def to_dict(self, rotations_type: str = "euler") -> Dict[str, Any]:
        """
        Serialize the ConstantAttitude to a dictionary.

        The dictionary format is as follows:
        {
            "rotations": list,
                # - If "rotations_type" is "quaternion": List of 4 values [x, y, z, w]
                # - If "rotations_type" is "euler": List of 3 values (Euler angles)
            "rotations_type": str,  # "quaternion" or "euler"
            "euler_order": str (optional),  # Only if rotations_type is "euler", default is "xyz"
            "base": str,  # Name of the base reference frame
        }

        Args:
            rotations_type (str): Either "quaternion" or "euler" (default: "euler").

        Returns:
            dict: Serialized ConstantAttitude.
        """
        if rotations_type == "quaternion":
            rotations = self.rotation.as_quat()
        elif rotations_type == "euler":
            rotations = self.rotation.as_euler("xyz").tolist()
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        result = {
            "rotations": rotations,
            "rotations_type": rotations_type,
            "base": self.base.to_string(),
        }
        if rotations_type == "euler":
            result["euler_order"] = "xyz"
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstantAttitude":
        """
        Construct a ConstantAttitude from a dictionary.

        The expected dictionary format is:
        {
            "rotations": list,
                # - If "rotations_type" is "quaternion": List of 4 values [x, y, z, w]
                # - If "rotations_type" is "euler": List of 3 values (Euler angles)
            "rotations_type": str,  # "quaternion" (default) or "euler"
            "euler_order": str (optional),  # Required if rotations_type is "euler"
            "base": str,  # Name of the base reference frame
        }

        Args:
            data (dict): Serialized ConstantAttitude dictionary.

        Returns:
            ConstantAttitude: Reconstructed object.

        Raises:
            ValueError: If Euler angles are used but no order is specified.
        """
        rotations_type = data.get("rotations_type", "quaternion")

        if rotations_type == "quaternion":
            rotation = Rotation.from_quat(data["rotations"])
        elif rotations_type == "euler":
            order = data.get("euler_order")
            if not order:
                raise ValueError(
                    "Missing required 'euler_order' for euler rotations."
                )
            rotation = Rotation.from_euler(order, data["rotations"])
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        base = ReferenceFrame.get(data["base"])
        return cls(rotation, base)


class SpiceAttitude(Attitude):
    """
    Attitude defined using SPICE frame transformations, including angular velocity.

    Attributes:
        from_frame (ReferenceFrame): Source frame for the SPICE transform.
        to_frame (ReferenceFrame): Target frame for the SPICE transform.
    """

    # Map ReferenceFrame to SPICE string names
    spice_names = {
        ReferenceFrame.ICRF_EC: "J2000",
        ReferenceFrame.ITRF: "ITRF93",
    }

    def __init__(self, from_frame: ReferenceFrame, to_frame: ReferenceFrame):
        self.from_frame = from_frame
        self.to_frame = to_frame

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray]
    ) -> tuple[Rotation, np.ndarray]:
        """
        Evaluate the SPICE attitude and angular velocity at the given time(s).

        Args:
            t (AbsoluteDate or AbsoluteDateArray): Time(s) for evaluation.

        Returns:
            Tuple[Rotation, np.ndarray]: Rotation(s) and angular velocity vector(s).
        """
        spice_from = self.spice_names[self.from_frame]
        spice_to = self.spice_names[self.to_frame]

        if isinstance(t, AbsoluteDate):
            sxform = spice.sxform(
                spice_from, spice_to, t.to_spice_ephemeris_time()
            )
            r_mat, w = spice.xf2rav(sxform)
            return Rotation.from_matrix(r_mat), np.array(w)
        elif isinstance(t, AbsoluteDateArray):
            et_array = t.ephemeris_time
            r_list = []
            w_list = []
            for et in et_array:
                sxform = spice.sxform(spice_from, spice_to, et)
                r_mat, w = spice.xf2rav(sxform)
                r_list.append(r_mat)
                w_list.append(w)
            return Rotation.from_matrix(np.array(r_list)), np.array(w_list)
        else:
            raise TypeError("t must be AbsoluteDate or AbsoluteDateArray")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a dictionary.

        Returns:
            dict: Dictionary with keys "from" and "to" (ReferenceFrame string names).
        """
        return {
            "from": self.from_frame.to_string(),
            "to": self.to_frame.to_string(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpiceAttitude":
        """
        Deserialize from a dictionary.

        Args:
            data (dict): Dictionary with "from" and "to" frame names.

        Returns:
            SpiceAttitude: Constructed object.
        """
        from_frame = ReferenceFrame.get(data["from"])
        to_frame = ReferenceFrame.get(data["to"])
        return cls(from_frame, to_frame)


class AttitudeSeries(Attitude):
    """
    Represents orientation data as a timeseries using scipy Rotation objects.

    Attributes:
        time (AbsoluteDateArray): Time samples.
        rotations (Rotation): Orientation at each time sample.
        base (ReferenceFrame): Base reference frame.
        angular_velocity (np.ndarray): Angular velocity vectors (Nx3).
    """

    def __init__(
        self,
        time: AbsoluteDateArray,
        rotations: Rotation,
        base: ReferenceFrame,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """
        Initialize an AttitudeSeries object.

        Args:
            time (AbsoluteDateArray): Array of time points.
            rotations (Rotation): Rotation object containing N orientations.
            base (ReferenceFrame): Static base reference frame.
            angular_velocity (np.ndarray, optional): Angular velocity vectors (Nx3).
                If None, computed automatically from rotations using finite difference.

        Raises:
            ValueError: If rotations length does not match time array length.
            ValueError: If angular_velocity shape is invalid.
        """
        if len(rotations) != len(time.ephemeris_time):
            raise ValueError("rotations and time must have the same length.")

        self.time = time
        self.rotations = rotations
        self.base = base

        if angular_velocity is not None:
            if angular_velocity.shape != (len(time.ephemeris_time), 3):
                raise ValueError("Angular_velocity must have shape (N, 3).")
            self.angular_velocity = angular_velocity
        else:
            self.angular_velocity = self._compute_angular_velocity()

    def _compute_angular_velocity(self) -> np.ndarray:
        """
        Compute angular velocity vectors from the rotation time series using central finite
        differences.

        For each internal sample t_i:
        1. Form the relative rotation from R_{i-1} to R_{i+1}:
            R_rel = R_{i-1}⁻¹ · R_{i+1}.
        2. Convert R_rel into its axis–angle vector v = θ·u, where
            - θ is the magnitude of rotation (in radians),
            - u is the unit axis of rotation,
            and SciPy’s `as_rotvec()` returns this v.
        3. Divide by the elapsed time Δt = t_{i+1} − t_{i−1} to approximate the instantaneous
        angular velocity:
            ω(t_i) ≈ v / Δt.

        Because this uses a central difference, the error scales with O(Δt²).  At the endpoints
        (i=0 and i=N−1) it falls back to simple forward/backward differences,
        which are only first-order accurate:
            ω(t₀) ≈ v₀ / (t₁ − t₀),
            ω(t_{N−1}) ≈ v_{N−1} / (t_{N−1} − t_{N−2}).

        Returns:
            np.ndarray: An (N×3) array of angular velocity vectors ω(t_i), expressed
                        in the body-fixed frame at each sample.
        """

        rotations = self.rotations
        t = self.time.ephemeris_time
        n = len(rotations)

        # Prepare output
        omega = np.zeros((n, 3))

        # 1) central differences for i=1…N-2 in vectorized operation:
        #    R_rel[i] = R_{i-1}⁻¹ * R_{i+1}  for i=1…N-2
        r_prev = rotations[:-2]
        r_next = rotations[2:]
        r_rel = r_prev.inv() * r_next  # shape (N-2,)
        v_rel = r_rel.as_rotvec()  # shape (N-2,3)
        dt = (t[2:] - t[:-2])[:, None]  # shape (N-2,1)
        omega[1:-1] = v_rel / dt

        # 2) endpoints
        forward = (rotations[0].inv() * rotations[1]).as_rotvec()
        backward = (rotations[-2].inv() * rotations[-1]).as_rotvec()
        omega[0] = forward / (t[1] - t[0])
        omega[-1] = backward / (t[-1] - t[-2])

        return omega

    def to_dict(self, rotations_type: str = "euler") -> Dict[str, Any]:
        """
        Serialize this AttitudeSeries into a dictionary.

        The dictionary format is as follows:
        {
            "time": dict,  # Serialized AbsoluteDateArray
            "rotations": list,  # List of rotations:
                                # - If "rotations_type" is "quaternion": List of quaternions (Nx4),
                                #   where each quaternion is [x, y, z, w] (scalar-last format).
                                # - If "rotations_type" is "euler": List of Euler angles (Nx3)
            "rotations_type": str,  # Type of rotations: "quaternion" or "euler"
            "euler_order": str (optional),  # Order (e.g., "xyz"), if "rotations_type" is "euler"
            "base": str,  # Base reference frame as a string identifier
            "angular_velocity": list  # List of angular velocity vectors (Nx3)
        }

        Args:
            rotations_type (str, optional): The type of rotations to serialize.
                Options are "quaternion" or "euler". Defaults to "euler".

        Returns:
            dict: Serialized representation of the AttitudeSeries.
        """
        if rotations_type == "quaternion":
            rotations = self.rotations.as_quat().tolist()
        elif rotations_type == "euler":
            rotations = self.rotations.as_euler(
                "xyz"
            ).tolist()  # Default to "xyz" order
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        result = {
            "time": self.time.to_dict(),
            "rotations": rotations,
            "rotations_type": rotations_type,
            "base": self.base.to_string(),
            "angular_velocity": self.angular_velocity.tolist(),
        }

        if rotations_type == "euler":
            result["euler_order"] = (
                "xyz"  # Include the Euler order if applicable
            )

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttitudeSeries":
        """
        Deserialize an AttitudeSeries from a dictionary.

        The expected dictionary format is:
        {
            "time": dict,  # Serialized AbsoluteDateArray
            "rotations": list,  # List of rotations:
                                # - If "rotations_type" is "quaternion": List of quaternions (Nx4),
                                #   where each quaternion is [x, y, z, w] (scalar-last format).
                                # - If "rotations_type" is "euler": List of Euler angles (Nx3)
            "rotations_type": str,  # Type of rotations: "quaternion" (default) or "euler"
            "euler_order": str (optional),  # Order (e.g., "xyz"), if "rotations_type" is "euler"
            "base": str,  # Base reference frame as a string identifier
            "angular_velocity": list (optional)  # List of angular velocity vectors (Nx3)
        }

        Args:
            data (dict): Serialized AttitudeSeries dictionary.

        Returns:
            AttitudeSeries: Reconstructed AttitudeSeries object.

        Raises:
            ValueError: If "rotations_type" is "euler" but "euler_order" is missing.
        """
        time = AbsoluteDateArray.from_dict(data["time"])
        rotations_type = data.get("rotations_type", "quaternion")

        if rotations_type == "quaternion":
            rotations = Rotation.from_quat(data["rotations"])
        elif rotations_type == "euler":
            euler_order = data.get("euler_order")
            if not euler_order:
                raise ValueError(
                    "euler_order is required when rotations_type is 'euler'."
                )
            rotations = Rotation.from_euler(euler_order, data["rotations"])
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        base = ReferenceFrame.get(
            data["base"]
        )  # Deserialize base as ReferenceFrame

        angular_velocity = (
            np.array(data["angular_velocity"])
            if "angular_velocity" in data
            else None
        )
        return cls(time, rotations, base, angular_velocity)

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray]
    ) -> tuple[Rotation, np.ndarray]:
        """
        Get the interpolated rotations and angular velocity at the specified time(s).

        Args:
            new_dates (AbsoluteDate or AbsoluteDateArray): Time(s) at which to evaluate.

        Returns:
            - If AbsoluteDate: a tuple (Rotation, angular_velocity_vector).
            - If AbsoluteDateArray: a tuple (Rotation, ndarray of angular velocities).
        """
        if isinstance(t, AbsoluteDate):
            new_et = np.array([t.ephemeris_time])
            rot, angvel = self._resample(new_et)
            return rot[0], angvel[0]
        elif isinstance(t, AbsoluteDateArray):
            return self._resample(t.ephemeris_time)
        else:
            raise TypeError(
                "new_dates must be an AbsoluteDate or AbsoluteDateArray."
            )

    def _resample(self, new_et: np.ndarray) -> tuple[Rotation, np.ndarray]:
        """
        Internal method to perform SLERP and angular velocity assignment.

        Args:
            new_et (np.ndarray): New ephemeris times (1D array).

        Returns:
            Tuple[Rotation, np.ndarray]: Interpolated rotations and angular velocity.
        """
        slerp = Slerp(self.time.ephemeris_time, self.rotations)
        new_rotations = slerp(new_et)

        indices = (
            np.searchsorted(self.time.ephemeris_time, new_et, side="right") - 1
        )
        indices = np.clip(indices, 0, len(self.angular_velocity) - 1)
        new_angular_velocity = self.angular_velocity[indices]

        return new_rotations, new_angular_velocity

    def resample(self, new_time: AbsoluteDateArray) -> "AttitudeSeries":
        """
        Resample AttitudeSeries to a new time base using spherical linear interpolation (SLERP).

        Args:
            new_time (AbsoluteDateArray): New time samples for interpolation.

        Returns:
            AttitudeSeries: New AttitudeSeries interpolated at the requested times.
        """
        new_rotations, new_angular_velocity = self._resample(
            new_time.ephemeris_time
        )
        return AttitudeSeries(
            time=new_time,
            rotations=new_rotations,
            base=self.base,
            angular_velocity=new_angular_velocity,
        )

    @classmethod
    def constant_velocity(
        cls,
        start_time: "AbsoluteDate",
        duration: float,
        initial_rotation: Rotation,
        angular_velocity: np.ndarray,
        base: ReferenceFrame,
    ) -> "AttitudeSeries":
        """
        Create an AttitudeSeries with a constant angular velocity.

        Args:
            start_time (AbsoluteDate): The starting time of the series.
            duration (float): Duration of the series in seconds.
            initial_rotation (Rotation): Initial orientation as a scipy Rotation object.
            angular_velocity (np.ndarray): Constant angular velocity vector (3D) in rad/s.
            base (ReferenceFrame): Base reference frame.

        Returns:
            AttitudeSeries: A new AttitudeSeries object.
        """
        if angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must be a 3D vector.")

        # Generate start and stop time samples
        time_samples = np.array(
            [
                start_time.ephemeris_time,
                start_time.ephemeris_time + duration,
            ]
        )
        time_array = AbsoluteDateArray(time_samples)

        # Compute rotations at start and stop times
        delta_rotation = Rotation.from_rotvec(angular_velocity * duration)
        rotations = Rotation.from_quat(
            [
                initial_rotation.as_quat(),
                (initial_rotation * delta_rotation).as_quat(),
            ]
        )

        # Return the new AttitudeSeries
        return cls(
            time=time_array,
            rotations=rotations,
            base=base,
            angular_velocity=np.tile(angular_velocity, (2, 1)),
        )

    @classmethod
    def get_lvlh(cls, state: StateSeries) -> "AttitudeSeries":
        """
        Compute an LVLH AttitudeSeries from a StateSeries in an inertial reference frame.

        Args:
            state (StateSeries): Trajectory in an inertial frame.

        Returns:
            AttitudeSeries: LVLH AttitudeSeries relative to state.frame.

        Raises:
            ValueError: If the state frame is not an inertial frame.
        """
        # Only support ICRF_EC as inertial
        if state.frame != ReferenceFrame.ICRF_EC:
            raise ValueError(
                f"LVLH only defined for inertial frames, got {state.frame}"
            )

        pos = state.data[0]  # shape (N,3)
        vel = state.data[1]  # shape (N,3)

        # Build direction cosine matrices
        r_mats = []
        for r, v in zip(pos, vel):
            r_norm = np.linalg.norm(r)
            if r_norm == 0:
                raise ValueError(
                    "Position vector has zero norm; cannot define LVLH axes"
                )
            x_hat = r / r_norm
            h = np.cross(r, v)
            h_norm = np.linalg.norm(h)
            if h_norm == 0:
                raise ValueError(
                    "Angular momentum vector has zero norm; cannot define LVLH axes"
                )
            z_hat = h / h_norm
            y_hat = np.cross(z_hat, x_hat)
            # Rows are LVLH axes in inertial coordinates
            r_mats.append(np.vstack([x_hat, y_hat, z_hat]))

        rotations = Rotation.from_matrix(np.array(r_mats))
        return cls(time=state.time, rotations=rotations, base=state.frame)

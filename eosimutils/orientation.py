"""
.. module:: eosimutils.orientation
   :synopsis: Classes for representing transformations between reference frames.
"""

import spiceypy as spice

from typing import Optional, Union, Dict, Any, Type, Callable
import numpy as np
from scipy.spatial.transform import Rotation as Scipy_Rotation
from scipy.spatial.transform import Slerp as Scipy_Slerp

from .base import ReferenceFrame
from .time import AbsoluteDate, AbsoluteDateArray
from .base import RotationsType


class Orientation:
    """
    Base class for orientation representations.

    Rotations are represented using SciPy rotation objects. A rotation object, when applied to a
    position or state (position and velocity) in from_frame, transforms the position (state) to
    to_frame. Rotation matrices are constructed from euler angles using the counterclockwise
    rotation convention (i.e., R(theta) is the matrix that rotates a vector counterclockwise by
    theta, see https://mathworld.wolfram.com/RotationMatrix.html. For instance, to construct an
    orientation object with from_frame A and to_frame B using Euler angles, the euler angles
    which rotate the B frame to the A frame would be provided.

    Subclasses must implement `at(t)` to return (Scipy_Rotation, angular_velocity). Note that
    Scipy Rotation objects use the scalar-last convention for quaternions. Counterclockwise
    rotations are positive.

    Implements a factory pattern to create instances from a dictionary using `from_dict(data)`.
    The dictionary must contain a "orientation_type" key to identify the subclass.

    Provides `transform(state, t)` to apply rotation and angular velocity to 6D state vectors.
    """

    # Registry for factory pattern
    _registry: Dict[str, Type["Orientation"]] = {}

    def __init__(self, from_frame: ReferenceFrame, to_frame: ReferenceFrame):
        self.from_frame = from_frame
        self.to_frame = to_frame

    @classmethod
    def register_type(
        cls, type_name: str
    ) -> Callable[[Type["Orientation"]], Type["Orientation"]]:
        """
        Decorator to register an Orientation subclass under a type name.
        """

        def decorator(subclass: Type["Orientation"]) -> Type["Orientation"]:
            cls._registry[type_name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Orientation":
        """
        Factory method to construct an Orientation from a serialized dictionary.
        Dispatches based on registered types.
        """
        orientation_type = data.get("orientation_type")
        subclass = cls._registry.get(orientation_type)
        if not subclass:
            raise ValueError(f"Unknown orientation type: {orientation_type}")
        return subclass.from_dict(data)

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray, None]
    ) -> tuple[Scipy_Rotation, np.ndarray]:
        """
        Return (rotation, angular_velocity) at the given time(s).

        The angular velocity is the angular velocity of from_frame w.r.t. to_frame,
        expressed in to_frame coordinates.

        Time can also be None for case of constant orientation.

        Args:
            t: AbsoluteDate or AbsoluteDateArray.
        Returns:
            Tuple[Scipy_Rotation, np.ndarray]: Scipy_Rotation and angular velocity vector(s).
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

    def inverse(self) -> "Orientation":
        """
        Return the inverse of the current orientation.

        Returns:
            Orientation: A new Orientation object with inverted frames and rotations.
        """
        raise NotImplementedError


@Orientation.register_type("constant")
class ConstantOrientation(Orientation):
    """
    Represents a time-invariant orientation using a single constant rotation.

    Attributes:
        rotation (Scipy_Rotation): The constant orientation.
        from_frame (ReferenceFrame): Source frame for the rotation.
        to_frame (ReferenceFrame): Target frame for the rotation.
    """

    def __init__(
        self,
        rotation: Scipy_Rotation,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
    ):
        """
        Initialize a ConstantOrientation.

        Args:
            rotation (Scipy_Rotation): A single Scipy_Rotation object.
            from_frame (ReferenceFrame): The source frame of the rotation.
            to_frame (ReferenceFrame): The target frame of the rotation.
        """
        super().__init__(from_frame, to_frame)
        self.rotation = rotation

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray, None]
    ) -> tuple[Scipy_Rotation, np.ndarray]:
        """
        Evaluate the orientation at the given time(s). Always returns the same rotation.

        Args:
            t (AbsoluteDate or AbsoluteDateArray): Input time(s).

        Returns:
            Tuple[Scipy_Rotation, np.ndarray]: The rotation and zero angular velocity vector.
        """
        if isinstance(t, (AbsoluteDate, type(None))):
            return self.rotation, np.zeros(3)
        elif isinstance(t, AbsoluteDateArray):
            n = len(t.ephemeris_time)
            return Scipy_Rotation.from_quat(
                [self.rotation.as_quat()] * n
            ), np.zeros((n, 3))
        else:
            raise TypeError("t must be AbsoluteDate or AbsoluteDateArray")

    def to_dict(
        self, rotations_type: Union[RotationsType, str] = RotationsType.EULER
    ) -> Dict[str, Any]:
        """
        Serialize the ConstantOrientation to a dictionary.

        The dictionary format is as follows:
        {
            "orientation_type": "constant",
            "rotations": list,
                # - If "rotations_type" is "quaternion": List of 4 values [x, y, z, w]
                # - If "rotations_type" is "euler": List of 3 values (Euler angles)
            "rotations_type": str,  # "quaternion" or "euler"
            "euler_order": str (optional),  # Order (e.g., "xyz"), if "rotations_type" is "euler".
                Use lowercase for intrinsic rotations, otherwise extrinsic. Defaults to "xyz".
            "from": str,  # Name of the source reference frame
            "to": str,  # Name of the target reference frame
        }

        Args:
            rotations_type (Union[RotationsType, str]): Either RotationsType.QUATERNION or
                RotationsType.EULER (default: RotationsType.EULER).

        Returns:
            dict: Serialized ConstantOrientation.
        """
        if isinstance(rotations_type, str):
            rotations_type = RotationsType.get(rotations_type)

        if rotations_type == RotationsType.QUATERNION:
            rotations = self.rotation.as_quat()
        elif rotations_type == RotationsType.EULER:
            rotations = self.rotation.as_euler("xyz").tolist()
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        result = {
            "orientation_type": "constant",
            "rotations": rotations,
            "rotations_type": rotations_type.to_string(),
            "from": self.from_frame.to_string(),
            "to": self.to_frame.to_string(),
        }
        if rotations_type == RotationsType.EULER:
            result["euler_order"] = "xyz"
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstantOrientation":
        """
        Construct a ConstantOrientation from a dictionary.

        The expected dictionary format is:
        {
            "rotations": list,
                # - If "rotations_type" is "quaternion": List of 4 values [x, y, z, w]
                # - If "rotations_type" is "euler": List of 3 values (Euler angles)
            "rotations_type": str,  # "quaternion" (default) or "euler"
            "euler_order": str (optional),  # Order (e.g., "xyz"), if "rotations_type" is "euler".
                Use lowercase for intrinsic rotations, otherwise extrinsic.
            "from": str,  # Name of the source reference frame
            "to": str,  # Name of the target reference frame
        }

        Args:
            data (dict): Serialized ConstantOrientation dictionary.

        Returns:
            ConstantOrientation: Reconstructed object.

        Raises:
            ValueError: If Euler angles are used but no order is specified.
        """
        rotations_type = RotationsType.get(
            data.get("rotations_type", "quaternion")
        )

        if rotations_type == RotationsType.QUATERNION:
            rotation = Scipy_Rotation.from_quat(data["rotations"])
        elif rotations_type == RotationsType.EULER:
            order = data.get("euler_order")
            if not order:
                raise ValueError(
                    "Missing required 'euler_order' for euler rotations."
                )
            rotation = Scipy_Rotation.from_euler(order, data["rotations"])
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        from_frame = ReferenceFrame.get(data["from"])
        to_frame = ReferenceFrame.get(data["to"])
        return cls(rotation, from_frame, to_frame)

    def inverse(self) -> "ConstantOrientation":
        """
        Return the inverse of the current constant orientation.

        Returns:
            ConstantOrientation: A new ConstantOrientation object with inverted rotation and frames.
        """
        return ConstantOrientation(
            self.rotation.inv(), self.to_frame, self.from_frame
        )


@Orientation.register_type("spice")
class SpiceOrientation(Orientation):
    """
    Orientation defined using SPICE frame transformations, including angular velocity.

    If there is no mapping from from_frame or to_frame to a spice frame name in
    the spice_names dictionary, the frame string is used directly. Hence, if
    the frame string is a valid spice frame name this function will work.

    Attributes:
        from_frame (ReferenceFrame): Source frame for the SPICE transform.
        to_frame (ReferenceFrame): Target frame for the SPICE transform.
        spice_names (dict): Mapping of ReferenceFrame to SPICE string names.
    """

    # Map ReferenceFrame to SPICE string names
    spice_names = {
        ReferenceFrame.get("ICRF_EC"): "J2000",
        ReferenceFrame.get("ITRF"): "ITRF93",
    }

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray]
    ) -> tuple[Scipy_Rotation, np.ndarray]:
        """
        Evaluate the SPICE orientation and angular velocity at the given time(s).

        Args:
            t (AbsoluteDate or AbsoluteDateArray): Time(s) for evaluation.

        Returns:
            Tuple[Scipy_Rotation, np.ndarray]: Scipy_Rotation(s) and angular velocity vector(s).
        """

        # If from_frame or to_frame is not in the spice_names, use the frame string directly
        # Spiceypy will throw an error if the frame string is not recognized as a
        # valid SPICE frame name.
        spice_from = self.spice_names.get(
            self.from_frame, self.from_frame.to_string()
        )
        spice_to = self.spice_names.get(
            self.to_frame, self.to_frame.to_string()
        )

        if isinstance(t, AbsoluteDate):
            sxform = spice.sxform(
                spice_from, spice_to, t.to_spice_ephemeris_time()
            )
            r_mat, w = spice.xf2rav(sxform)
            return Scipy_Rotation.from_matrix(r_mat), np.array(w)
        elif isinstance(t, AbsoluteDateArray):
            et_array = t.ephemeris_time
            r_list = []
            w_list = []
            for et in et_array:
                sxform = spice.sxform(spice_from, spice_to, et)
                r_mat, w = spice.xf2rav(sxform)
                r_list.append(r_mat)
                w_list.append(w)
            return Scipy_Rotation.from_matrix(np.array(r_list)), np.array(
                w_list
            )
        else:
            raise TypeError("t must be AbsoluteDate or AbsoluteDateArray")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a dictionary.

        Returns:
            dict: Dictionary with keys "from" and "to" (ReferenceFrame string names) and
            "orientation_type" : spice.
        """
        return {
            "orientation_type": "spice",
            "from": self.from_frame.to_string(),
            "to": self.to_frame.to_string(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpiceOrientation":
        """
        Deserialize from a dictionary.

        Args:
            data (dict): Dictionary with "from" and "to" frame names.

        Returns:
            SpiceOrientation: Constructed object.
        """
        from_frame = ReferenceFrame.get(data["from"])
        to_frame = ReferenceFrame.get(data["to"])
        return cls(from_frame, to_frame)

    def inverse(self) -> "SpiceOrientation":
        """
        Return the inverse of the current SPICE orientation.

        Returns:
            SpiceOrientation: A new SpiceOrientation object with inverted frames.
        """
        return SpiceOrientation(self.to_frame, self.from_frame)


@Orientation.register_type("series")
class OrientationSeries(Orientation):
    """
    Represents orientation data as a timeseries using scipy Scipy_Rotation objects.

    Attributes:
        time (AbsoluteDateArray): Time samples.
        rotations (Scipy_Rotation): Orientation at each time sample.
        from_frame (ReferenceFrame): Source frame for the rotations.
        to_frame (ReferenceFrame): Target frame for the rotations.
        angular_velocity (np.ndarray): Angular velocity vectors (Nx3) of from_frame wrt to_frame.
    """

    def __init__(
        self,
        time: AbsoluteDateArray,
        rotations: Scipy_Rotation,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """
        Initialize an OrientationSeries object.

        Args:
            time (AbsoluteDateArray): Array of time points.
            rotations (Scipy_Rotation): Scipy_Rotation object containing N orientations.
            from_frame (ReferenceFrame): Source frame for the rotations.
            to_frame (ReferenceFrame): Target frame for the rotations.
            angular_velocity (np.ndarray, optional): Angular velocity vectors (Nx3) of from_frame
                wrt to_frame. If None, computed automatically from rotations using finite
                difference.

        Raises:
            ValueError: If rotations length does not match time array length.
            ValueError: If angular_velocity shape is invalid.
        """
        if len(rotations) != len(time.ephemeris_time):
            raise ValueError("rotations and time must have the same length.")

        super().__init__(from_frame, to_frame)
        self.time = time
        self.rotations = rotations

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
            np.ndarray: An (N×3) array of angular velocity vectors ω(t_i)
                of from_frame wrt to_frame, expressed in to_frame coordinates.
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

    def to_dict(
        self,
        rotations_type: Union[RotationsType, str] = RotationsType.EULER,
        euler_order: str = "xyz",
    ) -> Dict[str, Any]:
        """
        Serialize this OrientationSeries into a dictionary.

        The dictionary format is as follows:
        {
            "orientation_type": "series",
            "time": dict,  # Serialized AbsoluteDateArray
            "rotations": list,  # List of rotations:
                                # - If "rotations_type" is "quaternion": List of quaternions (Nx4),
                                #   where each quaternion is [x, y, z, w] (scalar-last format).
                                # - If "rotations_type" is "euler": List of Euler angles (Nx3)
            "rotations_type": str,  # Type of rotations: "quaternion" or "euler"
            "euler_order": str (optional),  # Order (e.g., "xyz"), if "rotations_type" is "euler".
                Use lowercase for intrinsic rotations, otherwise extrinsic.
            "from": str,  # Source reference frame as a string identifier
            "to": str,  # Target reference frame as a string identifier
            "angular_velocity": list  # List of angular velocity vectors (Nx3)
        }

        Args:
            rotations_type (Union[RotationsType, str], optional): The type of rotations to
                serialize. Options are RotationsType.QUATERNION or RotationsType.EULER. Defaults
                to RotationsType.EULER.

        Returns:
            dict: Serialized representation of the OrientationSeries.
        """
        if isinstance(rotations_type, str):
            rotations_type = RotationsType.get(rotations_type)

        if rotations_type == RotationsType.QUATERNION:
            rotations = self.rotations.as_quat().tolist()
        elif rotations_type == RotationsType.EULER:
            rotations = self.rotations.as_euler(euler_order).tolist()
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        result = {
            "orientation_type": "series",
            "time": self.time.to_dict(),
            "rotations": rotations,
            "rotations_type": rotations_type.to_string(),
            "from": self.from_frame.to_string(),
            "to": self.to_frame.to_string(),
            "angular_velocity": self.angular_velocity.tolist(),
        }

        if rotations_type == RotationsType.EULER:
            result["euler_order"] = euler_order

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrientationSeries":
        """
        Deserialize an OrientationSeries from a dictionary.

        The expected dictionary format is:
        {
            "time": dict,  # Serialized AbsoluteDateArray
            "rotations": list,  # List of rotations:
                                # - If "rotations_type" is "quaternion": List of quaternions (Nx4),
                                #   where each quaternion is [x, y, z, w] (scalar-last format).
                                # - If "rotations_type" is "euler": List of Euler angles (Nx3)
            "rotations_type": str,  # Type of rotations: "quaternion" (default) or "euler"
            "euler_order": str (optional),  # Order (e.g., "xyz"), if "rotations_type" is "euler".
                Use lowercase for intrinsic rotations, otherwise extrinsic.
            If coordinate axes are lowercase, uses intrinsic rotations, otherwise extrinsic.
            "from": str,  # Source reference frame as a string identifier
            "to": str,  # Target reference frame as a string identifier
            "angular_velocity": list (optional)  # List of angular velocity vectors (Nx3) of
                from_frame wrt to_frame
        }

        Args:
            data (dict): Serialized OrientationSeries dictionary.

        Returns:
            OrientationSeries: Reconstructed OrientationSeries object.

        Raises:
            ValueError: If "rotations_type" is "euler" but "euler_order" is missing.
        """
        time = AbsoluteDateArray.from_dict(data["time"])
        rotations_type = RotationsType.get(
            data.get("rotations_type", "quaternion")
        )

        if rotations_type == RotationsType.QUATERNION:
            rotations = Scipy_Rotation.from_quat(data["rotations"])
        elif rotations_type == RotationsType.EULER:
            euler_order = data.get("euler_order")
            if not euler_order:
                raise ValueError(
                    "euler_order is required when rotations_type is 'euler'."
                )
            rotations = Scipy_Rotation.from_euler(
                euler_order, data["rotations"]
            )
        else:
            raise ValueError(f"Unsupported rotations_type: {rotations_type}")

        from_frame = ReferenceFrame.get(data["from"])
        to_frame = ReferenceFrame.get(data["to"])

        angular_velocity = (
            np.array(data["angular_velocity"])
            if "angular_velocity" in data
            else None
        )
        return cls(time, rotations, from_frame, to_frame, angular_velocity)

    def at(
        self, t: Union[AbsoluteDate, AbsoluteDateArray]
    ) -> tuple[Scipy_Rotation, np.ndarray]:
        """
        Get the interpolated rotations and angular velocity at the specified time(s).

        Args:
            new_dates (AbsoluteDate or AbsoluteDateArray): Time(s) at which to evaluate.

        Returns:
            - If AbsoluteDate: a tuple (Scipy_Rotation, angular_velocity_vector).
            - If AbsoluteDateArray: a tuple (Scipy_Rotation, ndarray of angular velocities).
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

    def _resample(
        self, new_et: np.ndarray
    ) -> tuple[Scipy_Rotation, np.ndarray]:
        """
        Internal method to perform SLERP and angular velocity assignment.

        Args:
            new_et (np.ndarray): New ephemeris times (1D array).

        Returns:
            Tuple[Scipy_Rotation, np.ndarray]: Interpolated rotations and angular velocity.
        """
        slerp = Scipy_Slerp(self.time.ephemeris_time, self.rotations)
        new_rotations = slerp(new_et)

        indices = (
            np.searchsorted(self.time.ephemeris_time, new_et, side="right") - 1
        )
        indices = np.clip(indices, 0, len(self.angular_velocity) - 1)
        new_angular_velocity = self.angular_velocity[indices]

        return new_rotations, new_angular_velocity

    def resample(self, new_time: AbsoluteDateArray) -> "OrientationSeries":
        """
        Resample OrientationSeries to a new time base using spherical linear interpolation (SLERP).

        Args:
            new_time (AbsoluteDateArray): New time samples for interpolation.

        Returns:
            OrientationSeries: New OrientationSeries interpolated at the requested times.
        """
        new_rotations, new_angular_velocity = self._resample(
            new_time.ephemeris_time
        )
        return OrientationSeries(
            time=new_time,
            rotations=new_rotations,
            from_frame=self.from_frame,
            to_frame=self.to_frame,
            angular_velocity=new_angular_velocity,
        )

    @classmethod
    def constant_velocity(
        cls,
        start_time: "AbsoluteDate",
        duration: float,
        initial_rotation: Scipy_Rotation,
        angular_velocity: np.ndarray,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
    ) -> "OrientationSeries":
        """
        Create an OrientationSeries with a constant angular velocity.

        Args:
            start_time (AbsoluteDate): The starting time of the series.
            duration (float): Duration of the series in seconds.
            initial_rotation (Scipy_Rotation): Initial orientation as a scipy Scipy_Rotation object.
            angular_velocity (np.ndarray): Constant angular velocity vector (3D) in rad/s.
            from_frame (ReferenceFrame): Source frame for the rotations.
            to_frame (ReferenceFrame): Target frame for the rotations.
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
        delta_rotation = Scipy_Rotation.from_rotvec(angular_velocity * duration)
        rotations = Scipy_Rotation.from_quat(
            [
                initial_rotation.as_quat(),
                (initial_rotation * delta_rotation).as_quat(),
            ]
        )

        # Return the new OrientationSeries
        return cls(
            time=time_array,
            rotations=rotations,
            from_frame=from_frame,
            to_frame=to_frame,
            angular_velocity=np.tile(angular_velocity, (2, 1)),
        )

    def inverse(self) -> "OrientationSeries":
        """
        Return the inverse of the current orientation series.

        Returns:
            OrientationSeries: A new OrientationSeries object with inverted rotations and frames.
        """
        return OrientationSeries(
            time=self.time,
            rotations=self.rotations.inv(),
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            angular_velocity=-self.angular_velocity,
        )

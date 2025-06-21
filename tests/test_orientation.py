"""Unit tests for the eosimutils.orientation module.

Tests for the `SpiceOrientation` class are located in the
`test_orientation_spiceorientation` script.
"""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation as Scipy_Rotation
from eosimutils.orientation import (
    ConstantOrientation,
    OrientationSeries,
    Orientation,
    SpiceOrientation,
)
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.base import ReferenceFrame


class TestConstantOrientation(unittest.TestCase):
    """Test the ConstantOrientation class."""

    # Construct 90 deg rotation about Z axis
    # and use it to build a ConstantOrientation object
    # using from_dict with quaternion serialization
    def setUp(self):
        # Use built-in frames
        self.frm = ReferenceFrame.get("ICRF_EC")
        self.to = ReferenceFrame.get("ITRF")
        # 90 deg rotation about Z axis
        self.rotation = Scipy_Rotation.from_euler(
            "xyz", [0.0, 0.0, 90.0], degrees=True
        )

        d = {
            "rotations": self.rotation.as_quat().tolist(),
            "rotations_type": "quaternion",
            "from": self.frm.to_string(),
            "to": self.to.to_string(),
        }
        self.co = ConstantOrientation.from_dict(d)

    def test_at_returns_constant_rotation_and_zero_angular_velocity(self):
        # at(None) should return the same rotation and a zero vector
        rot, omega = self.co.at(None)
        np.testing.assert_allclose(
            rot.as_euler("xyz"), self.rotation.as_euler("xyz"), atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

        # at a single AbsoluteDate
        t0 = AbsoluteDate(0.0)
        rot1, omega1 = self.co.at(t0)
        np.testing.assert_allclose(
            rot1.as_euler("xyz"), self.rotation.as_euler("xyz"), atol=1e-8
        )
        np.testing.assert_array_equal(omega1, np.zeros(3))

        # at an AbsoluteDateArray of length N
        et = np.array([0.0, 1.0, 2.0])
        t_arr = AbsoluteDateArray(et)
        rot_arr, omega_arr = self.co.at(t_arr)
        self.assertEqual(len(rot_arr), 3)
        self.assertEqual(omega_arr.shape, (3, 3))
        actual_eulers = rot_arr.as_euler("xyz")
        expected_eulers = [self.rotation.as_euler("xyz")] * len(actual_eulers)
        for act, exp in zip(actual_eulers, expected_eulers):
            np.testing.assert_allclose(act, exp, atol=1e-8)
        np.testing.assert_array_equal(omega_arr, np.zeros((3, 3)))

    def test_inverse_swaps_frames_and_inverts_rotation(self):
        # inverse() should swap from/to and invert the rotation
        inv = self.co.inverse()
        self.assertEqual(inv.from_frame, self.to)
        self.assertEqual(inv.to_frame, self.frm)
        np.testing.assert_allclose(
            inv.rotation.as_euler("xyz"),
            self.rotation.inv().as_euler("xyz"),
            atol=1e-8,
        )

    def test_to_dict_and_from_dict_roundtrip(self):
        # Quaternion‐based serialization
        d_q = self.co.to_dict(rotations_type="quaternion")
        co_q = ConstantOrientation.from_dict(d_q)
        np.testing.assert_allclose(
            co_q.rotation.as_euler("xyz"),
            self.rotation.as_euler("xyz"),
            atol=1e-8,
        )
        self.assertEqual(co_q.from_frame, self.frm)
        self.assertEqual(co_q.to_frame, self.to)

        # Euler‐based serialization
        d_e = self.co.to_dict(rotations_type="euler")
        co_e = ConstantOrientation.from_dict(d_e)
        np.testing.assert_allclose(
            co_e.rotation.as_euler("xyz"),
            self.rotation.as_euler("xyz"),
            atol=1e-8,
        )
        self.assertEqual(co_e.from_frame, self.frm)
        self.assertEqual(co_e.to_frame, self.to)


class TestOrientationSeries(unittest.TestCase):
    """Test the OrientationSeries class."""

    def setUp(self):
        # Define a constant angular velocity about Z: omega = pi/2 rad/s
        self.omega = np.array([0.0, 0.0, np.pi / 2])
        # Time samples t = [0, 1, 2]
        et = np.array([0.0, 1.0, 2.0])
        self.times = AbsoluteDateArray(et)

        # Euler angles for pure Z‐axis rotations: [0, 0, omega t]
        euler_angles = [[0.0, 0.0, self.omega[2] * t] for t in et]
        # Precompute Scipy rotations for test comparisons
        self.rotations = Scipy_Rotation.from_euler("xyz", euler_angles)

        # Use built‐in frames
        self.frm = ReferenceFrame.get("ICRF_EC")
        self.to = ReferenceFrame.get("ITRF")

        # Construct series via from_dict
        data_dict = {
            "time": self.times.to_dict(),
            "rotations": euler_angles,
            "rotations_type": "euler",
            "euler_order": "xyz",
            "from": self.frm.to_string(),
            "to": self.to.to_string(),
        }
        # angular_velocity omitted so it will be computed internally
        self.series = OrientationSeries.from_dict(data_dict)

    def test_computed_angular_velocity_is_constant(self):
        # Computed omega at each sample should match the known omega
        av = self.series.angular_velocity
        self.assertEqual(av.shape, (3, 3))
        for w in av:
            np.testing.assert_allclose(w, self.omega, atol=1e-8)

    def test_at_method_returns_correct_rotation_and_velocity(self):
        # at(t=1) should return the second rotation and omega
        t1 = AbsoluteDate(1.0)
        rot1, w1 = self.series.at(t1)
        # Fails if tolerance is one order of magnitude lower
        np.testing.assert_allclose(
            rot1.as_euler("xyz"),
            self.rotations[1].as_euler("xyz"),
            rtol=1e-5,
            atol=1e-4,
        )
        np.testing.assert_allclose(w1, self.omega, atol=1e-8)

        # at multiple times
        t_arr = AbsoluteDateArray(np.array([0.5, 1.5]))
        rot_arr, w_arr = self.series.at(t_arr)
        # rotations via SLERP: half and three-halves of full step
        expected_eulers = [[0.0, 0.0, self.omega[2] * t] for t in [0.5, 1.5]]
        actual_eulers = rot_arr.as_euler("xyz")

        # Fails if tolerance is one order of magnitude lower
        np.testing.assert_allclose(
            actual_eulers, expected_eulers, rtol=1e-3, atol=1e-3
        )
        # angular velocity picks nearest sample's omega
        for w in w_arr:
            np.testing.assert_allclose(w, self.omega, atol=1e-8)

    def test_resample_interpolates_and_preserves_omega(self):
        # Resample at midpoints [0.5, 1.5]
        new_et = np.array([0.5, 1.5])
        new_time = AbsoluteDateArray(new_et)
        resampled = self.series.resample(new_time)
        # Check rotations via Euler angles
        expected_eulers = [[0.0, 0.0, self.omega[2] * t] for t in new_et]
        actual_eulers = resampled.rotations.as_euler("xyz")
        np.testing.assert_allclose(
            actual_eulers, expected_eulers, rtol=1e-4, atol=1e-4
        )
        # Check angular velocities
        for w in resampled.angular_velocity:
            np.testing.assert_allclose(w, self.omega, atol=1e-8)

    def test_to_dict_and_from_dict_roundtrip(self):
        # Serialize with Euler angles
        d = self.series.to_dict(rotations_type="euler", euler_order="xyz")
        s2 = OrientationSeries.from_dict(d)

        # Rotations should match
        np.testing.assert_allclose(
            s2.rotations.as_euler("xyz"),
            self.series.rotations.as_euler("xyz"),
            atol=1e-8,
        )
        # Angular velocities should match
        np.testing.assert_allclose(
            s2.angular_velocity, self.series.angular_velocity, atol=1e-8
        )
        # Frames should match
        self.assertEqual(s2.from_frame, self.frm)
        self.assertEqual(s2.to_frame, self.to)


class TestOrientation(unittest.TestCase):
    """Test the factory pattern for Orientation subclasses."""

    def test_constant_orientation_from_dict(self):
        # Create a ConstantOrientation and serialize it
        frm = ReferenceFrame.get("ICRF_EC")
        to = ReferenceFrame.get("ITRF")
        rotation = Scipy_Rotation.from_euler(
            "xyz", [0.0, 0.0, 90.0], degrees=True
        )
        co = ConstantOrientation(rotation, frm, to)
        data = co.to_dict(rotations_type="quaternion")

        # Deserialize using the factory method
        co_from_dict = Orientation.from_dict(data)
        self.assertIsInstance(co_from_dict, ConstantOrientation)
        np.testing.assert_allclose(
            co_from_dict.rotation.as_quat(), rotation.as_quat(), atol=1e-8
        )
        self.assertEqual(co_from_dict.from_frame, frm)
        self.assertEqual(co_from_dict.to_frame, to)

    def test_orientation_series_from_dict(self):
        # Create an OrientationSeries and serialize it
        frm = ReferenceFrame.get("ICRF_EC")
        to = ReferenceFrame.get("ITRF")
        times = AbsoluteDateArray(np.array([0.0, 1.0, 2.0]))
        rotations = Scipy_Rotation.from_euler(
            "xyz",
            [[0.0, 0.0, 0.0], [0.0, 0.0, 90.0], [0.0, 0.0, 180.0]],
            degrees=True,
        )
        angular_velocity = np.array([[0.0, 0.0, np.pi / 2]] * 3)
        series = OrientationSeries(times, rotations, frm, to, angular_velocity)
        data = series.to_dict(rotations_type="euler", euler_order="xyz")

        # Deserialize using the factory method
        series_from_dict = Orientation.from_dict(data)
        self.assertIsInstance(series_from_dict, OrientationSeries)
        np.testing.assert_allclose(
            series_from_dict.rotations.as_euler("xyz"),
            rotations.as_euler("xyz"),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            series_from_dict.angular_velocity, angular_velocity, atol=1e-8
        )
        self.assertEqual(series_from_dict.from_frame, frm)
        self.assertEqual(series_from_dict.to_frame, to)

    def test_spice_orientation_from_dict(self):
        # Create a SpiceOrientation and serialize it
        frm = ReferenceFrame.get("ICRF_EC")
        to = ReferenceFrame.get("ITRF")
        spice_orientation = SpiceOrientation(frm, to)
        data = spice_orientation.to_dict()

        # Deserialize using the factory method
        spice_from_dict = Orientation.from_dict(data)
        self.assertIsInstance(spice_from_dict, SpiceOrientation)
        self.assertEqual(spice_from_dict.from_frame, frm)
        self.assertEqual(spice_from_dict.to_frame, to)


if __name__ == "__main__":
    unittest.main()

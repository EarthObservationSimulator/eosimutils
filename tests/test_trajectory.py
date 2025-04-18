"""Unit tests for eosimutils.trajectory module."""

import unittest
import numpy as np
from eosimutils.trajectory import StateSeries, PositionSeries
from eosimutils.time import AbsoluteDateArray, AbsoluteDate, JD_OF_J2000
from eosimutils.base import ReferenceFrame
from eosimutils.state import (
    CartesianState,
    Cartesian3DPosition,
    Cartesian3DVelocity,
)


class TestStateSeries(unittest.TestCase):
    """Unit tests for the StateSeries class."""

    def setUp(self):
        self.time = AbsoluteDateArray.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": [JD_OF_J2000 + t for t in [0, 1, 2]],
                "time_scale": "UTC",
            }
        )
        self.positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.velocities = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.frame = ReferenceFrame("ICRF_EC")
        self.trajectory = StateSeries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [self.positions.tolist(), self.velocities.tolist()],
                "frame": self.frame.to_string(),
                "headers": [
                    ["pos_x", "pos_y", "pos_z"],
                    ["vel_x", "vel_y", "vel_z"],
                ],
            }
        )

    def test_resample(self):
        """Test resampling of trajectory data."""
        new_time = np.array([JD_OF_J2000 + t for t in [0.5, 1.5]])
        new_time_obj = AbsoluteDateArray.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": new_time.tolist(),
                "time_scale": "UTC",
            }
        )
        resampled = self.trajectory.resample(new_time_obj)
        self.assertEqual(
            resampled.time.ephemeris_time.tolist(),
            new_time_obj.ephemeris_time.tolist(),
        )
        np.testing.assert_allclose(
            resampled.data[0], [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
        )
        np.testing.assert_allclose(
            resampled.data[1], [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
        )

    def test_remove_gaps(self):
        """Test removal of gaps in trajectory data."""
        positions_with_gaps = np.array(
            [[np.nan, np.nan, np.nan], [1, 1, 1], [np.nan, np.nan, np.nan]]
        )
        velocities_with_gaps = np.array(
            [[np.nan, np.nan, np.nan], [1, 1, 1], [np.nan, np.nan, np.nan]]
        )
        trajectory_with_gaps = StateSeries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [
                    positions_with_gaps.tolist(),
                    velocities_with_gaps.tolist(),
                ],
                "frame": self.frame.to_string(),
                "headers": [
                    ["pos_x", "pos_y", "pos_z"],
                    ["vel_x", "vel_y", "vel_z"],
                ],
            }
        )
        trimmed = trajectory_with_gaps.remove_gaps()
        np.testing.assert_array_equal(
            trimmed.time.to_dict("JULIAN_DATE")["jd"], [JD_OF_J2000 + 1]
        )
        np.testing.assert_array_equal(trimmed.data[0], [[1, 1, 1]])
        np.testing.assert_array_equal(trimmed.data[1], [[1, 1, 1]])

    def test_to_frame(self):
        """Test conversion of trajectory to a different reference frame."""
        to_frame = ReferenceFrame("ITRF")
        converted = self.trajectory.to_frame(to_frame)
        self.assertEqual(converted.frame, to_frame)
        # Assuming convert_frame is tested separately, we only check the frame here.

    def test_arithmetic_operations(self):
        """Test arithmetic operations between trajectories."""
        other_positions = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        other_velocities = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        other_trajectory = StateSeries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [other_positions.tolist(), other_velocities.tolist()],
                "frame": self.frame.to_string(),
                "headers": [
                    ["pos_x", "pos_y", "pos_z"],
                    ["vel_x", "vel_y", "vel_z"],
                ],
            }
        )

        added = self.trajectory + other_trajectory
        np.testing.assert_array_equal(
            added.data[0], self.positions + other_positions
        )
        np.testing.assert_array_equal(
            added.data[1], self.velocities + other_velocities
        )

        subtracted = self.trajectory - other_trajectory
        np.testing.assert_array_equal(
            subtracted.data[0], self.positions - other_positions
        )
        np.testing.assert_array_equal(
            subtracted.data[1], self.velocities - other_velocities
        )

    def test_constant_position(self):
        """Test creation of a constant position trajectory."""
        position = np.array([1, 1, 1])
        constant_traj = StateSeries.constant_position(
            JD_OF_J2000 + 0, JD_OF_J2000 + 1, position, self.frame
        )
        np.testing.assert_array_equal(
            constant_traj.data[0], [[1, 1, 1], [1, 1, 1]]
        )
        np.testing.assert_array_equal(
            constant_traj.data[1], [[0, 0, 0], [0, 0, 0]]
        )

    def test_constant_velocity(self):
        """Test creation of a constant velocity trajectory."""
        initial_position = np.array([0, 0, 0])
        velocity = np.array([1, 1, 1])
        constant_vel_traj = StateSeries.constant_velocity(
            JD_OF_J2000 + 0,
            JD_OF_J2000 + 1,
            velocity,
            initial_position,
            self.frame,
        )
        np.testing.assert_array_equal(
            constant_vel_traj.data[0], [[0, 0, 0], [1, 1, 1]]
        )
        np.testing.assert_array_equal(
            constant_vel_traj.data[1], [[1, 1, 1], [1, 1, 1]]
        )

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of StateSeries."""
        serialized = self.trajectory.to_dict()
        deserialized = StateSeries.from_dict(serialized)

        # All close used due to rounding in serialization.
        # TODO: May be worth looking into precision of timestrings in the future.
        np.testing.assert_allclose(
            deserialized.time.ephemeris_time,
            self.trajectory.time.ephemeris_time,
            rtol=1e-4,
            atol=1e-4,
        )
        for d1, d2 in zip(deserialized.data, self.trajectory.data):
            np.testing.assert_array_equal(d1, d2)
        self.assertEqual(deserialized.frame, self.trajectory.frame)

    def test_from_list_of_cartesian_state(self):
        """Test creation of a StateSeries from a list of CartesianState objects."""
        frame = ReferenceFrame("ICRF_EC")
        states = [
            CartesianState(
                time=AbsoluteDate(JD_OF_J2000 + 0),
                position=Cartesian3DPosition(0, 0, 0, frame),
                velocity=Cartesian3DVelocity(1, 1, 1, frame),
                frame=frame,
            ),
            CartesianState(
                time=AbsoluteDate(JD_OF_J2000 + 1),
                position=Cartesian3DPosition(1, 1, 1, frame),
                velocity=Cartesian3DVelocity(2, 2, 2, frame),
                frame=frame,
            ),
            CartesianState(
                time=AbsoluteDate(JD_OF_J2000 + 2),
                position=Cartesian3DPosition(2, 2, 2, frame),
                velocity=Cartesian3DVelocity(3, 3, 3, frame),
                frame=frame,
            ),
        ]
        trajectory = StateSeries.from_list_of_cartesian_state(states)
        np.testing.assert_array_equal(
            trajectory.time.ephemeris_time, [JD_OF_J2000 + t for t in [0, 1, 2]]
        )
        np.testing.assert_array_equal(
            trajectory.data[0], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        )
        np.testing.assert_array_equal(
            trajectory.data[1], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        )
        self.assertEqual(trajectory.frame, frame)

        # Test for mismatched frames
        states[1] = CartesianState(
            time=AbsoluteDate(JD_OF_J2000 + 1),
            position=Cartesian3DPosition(1, 1, 1, ReferenceFrame("ITRF")),
            velocity=Cartesian3DVelocity(2, 2, 2, ReferenceFrame("ITRF")),
            frame=ReferenceFrame("ITRF"),
        )
        with self.assertRaises(ValueError):
            StateSeries.from_list_of_cartesian_state(states)


class TestPositionSeries:
    """Unit tests for the PositionSeries class."""

    def test_initialization(self):
        """Test initialization of PositionSeries."""
        time = AbsoluteDateArray.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": [JD_OF_J2000 + t for t in [0.0, 1.0]],
                "time_scale": "UTC",
            }
        )
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frame = ReferenceFrame.ICRF_EC
        ps = PositionSeries.from_dict(
            {
                "time": time.to_dict("JULIAN_DATE"),
                "data": data.tolist(),
                "frame": frame.to_string(),
                "headers": ["pos_x", "pos_y", "pos_z"],
            }
        )
        assert np.array_equal(ps.data[0], data)
        assert ps.frame == frame

    def test_resample(self):
        """Test resampling of PositionSeries."""
        time = AbsoluteDateArray(
            np.array([JD_OF_J2000 + t for t in [0.0, 1.0]])
        )
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frame = ReferenceFrame.ICRF_EC
        ps = PositionSeries(time, data, frame)
        new_time = np.array([JD_OF_J2000 + 0.5])
        resampled_ps = ps.resample(new_time)
        assert resampled_ps.data[0].shape == (1, 3)

    def test_remove_gaps(self):
        """Test removing gaps (NaN values) from PositionSeries."""
        time = AbsoluteDateArray(
            np.array([JD_OF_J2000 + t for t in [0.0, 1.0, 2.0]])
        )
        data = np.array(
            [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]]
        )
        frame = ReferenceFrame.ICRF_EC
        ps = PositionSeries(time, data, frame)
        gapless_ps = ps.remove_gaps()
        assert gapless_ps.data[0].shape == (2, 3)

    def test_to_frame(self):
        """Test frame conversion for PositionSeries."""
        time = AbsoluteDateArray(
            np.array([JD_OF_J2000 + t for t in [0.0, 1.0]])
        )
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frame = ReferenceFrame.ICRF_EC
        ps = PositionSeries(time, data, frame)
        converted_ps = ps.to_frame(ReferenceFrame.ITRF)
        assert converted_ps.frame == ReferenceFrame.ITRF

    def test_from_list_of_cartesian_position(self):
        """Test creation of PositionSeries from a list of Cartesian3DPosition objects."""
        positions = [
            Cartesian3DPosition(1.0, 2.0, 3.0, ReferenceFrame.ICRF_EC),
            Cartesian3DPosition(4.0, 5.0, 6.0, ReferenceFrame.ICRF_EC),
        ]
        for pos in positions:
            pos.time = AbsoluteDateArray(
                np.array([JD_OF_J2000 + t for t in [0.0, 1.0]])
            )  # Mock time attribute
        ps = PositionSeries.from_list_of_cartesian_position(positions)
        assert ps.data[0].shape == (2, 3)
        assert ps.frame == ReferenceFrame.ICRF_EC

    def test_arithmetic_operations(self):
        """Test arithmetic operations between PositionSeries."""
        time = AbsoluteDateArray(
            np.array([JD_OF_J2000 + t for t in [0.0, 1.0, 2.0]])
        )
        data1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        data2 = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5]])
        frame = ReferenceFrame.ICRF_EC

        ps1 = PositionSeries(time, data1, frame)
        ps2 = PositionSeries(time, data2, frame)

        # Test addition
        added = ps1 + ps2
        np.testing.assert_array_equal(added.data[0], data1 + data2)

        # Test subtraction
        subtracted = ps1 - ps2
        np.testing.assert_array_equal(subtracted.data[0], data1 - data2)

        # Test multiplication
        multiplied = ps1 * 2
        np.testing.assert_array_equal(multiplied.data[0], data1 * 2)

        # Test division
        divided = ps1 / 2
        np.testing.assert_array_equal(divided.data[0], data1 / 2)

        # Test scalar addition
        scalar_added = ps1 + 1
        np.testing.assert_array_equal(scalar_added.data[0], data1 + 1)

        # Test scalar subtraction
        scalar_subtracted = ps1 - 1
        np.testing.assert_array_equal(scalar_subtracted.data[0], data1 - 1)


if __name__ == "__main__":
    unittest.main()

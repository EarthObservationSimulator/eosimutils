"""Unit tests for eosimutils.trajectory module."""

import unittest
import numpy as np
from eosimutils.trajectory import Trajectory, PositionSeries
from eosimutils.time import AbsoluteDateArray, AbsoluteDate
from eosimutils.base import ReferenceFrame
from eosimutils.state import (
    CartesianState,
    Cartesian3DPosition,
    Cartesian3DVelocity,
)


class TestTrajectory(unittest.TestCase):
    """Unit tests for the Trajectory class."""

    def setUp(self):
        self.time = AbsoluteDateArray(np.array([0, 1, 2]))
        self.positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.velocities = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.frame = ReferenceFrame("ICRF_EC")
        self.trajectory = Trajectory(
            self.time, [self.positions, self.velocities], self.frame
        )

    def test_resample(self):
        """Test resampling of trajectory data."""
        new_time = np.array([0.5, 1.5])
        resampled = self.trajectory.resample(new_time)
        self.assertEqual(resampled.time.et.tolist(), new_time.tolist())
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
        trajectory_with_gaps = Trajectory(
            self.time, [positions_with_gaps, velocities_with_gaps], self.frame
        )
        trimmed = trajectory_with_gaps.remove_gaps()
        np.testing.assert_allclose(trimmed.time.et, [1])
        np.testing.assert_allclose(trimmed.data[0], [[1, 1, 1]])
        np.testing.assert_allclose(trimmed.data[1], [[1, 1, 1]])

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
        other_trajectory = Trajectory(
            self.time, [other_positions, other_velocities], self.frame
        )

        added = self.trajectory + other_trajectory
        np.testing.assert_allclose(
            added.data[0], self.positions + other_positions
        )
        np.testing.assert_allclose(
            added.data[1], self.velocities + other_velocities
        )

        subtracted = self.trajectory - other_trajectory
        np.testing.assert_allclose(
            subtracted.data[0], self.positions - other_positions
        )
        np.testing.assert_allclose(
            subtracted.data[1], self.velocities - other_velocities
        )

    def test_constant_position(self):
        """Test creation of a constant position trajectory."""
        position = np.array([1, 1, 1])
        constant_traj = Trajectory.constant_position(0, 1, position, self.frame)
        np.testing.assert_allclose(
            constant_traj.data[0], [[1, 1, 1], [1, 1, 1]]
        )
        np.testing.assert_allclose(
            constant_traj.data[1], [[0, 0, 0], [0, 0, 0]]
        )

    def test_constant_velocity(self):
        """Test creation of a constant velocity trajectory."""
        initial_position = np.array([0, 0, 0])
        velocity = np.array([1, 1, 1])
        constant_vel_traj = Trajectory.constant_velocity(
            0, 1, velocity, initial_position, self.frame
        )
        np.testing.assert_allclose(
            constant_vel_traj.data[0], [[0, 0, 0], [1, 1, 1]]
        )
        np.testing.assert_allclose(
            constant_vel_traj.data[1], [[1, 1, 1], [1, 1, 1]]
        )

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of Trajectory."""
        serialized = self.trajectory.to_dict()
        deserialized = Trajectory.from_dict(serialized)
        np.testing.assert_allclose(
            deserialized.time.et, self.trajectory.time.et, rtol=1e-4, atol=1e-4
        )
        for d1, d2 in zip(deserialized.data, self.trajectory.data):
            np.testing.assert_allclose(d1, d2)
        self.assertEqual(deserialized.frame, self.trajectory.frame)

    def test_from_list_of_cartesian_state(self):
        """Test creation of a Trajectory from a list of CartesianState objects."""
        frame = ReferenceFrame("ICRF_EC")
        states = [
            CartesianState(
                time=AbsoluteDate(0),
                position=Cartesian3DPosition(0, 0, 0, frame),
                velocity=Cartesian3DVelocity(1, 1, 1, frame),
                frame=frame,
            ),
            CartesianState(
                time=AbsoluteDate(1),
                position=Cartesian3DPosition(1, 1, 1, frame),
                velocity=Cartesian3DVelocity(2, 2, 2, frame),
                frame=frame,
            ),
            CartesianState(
                time=AbsoluteDate(2),
                position=Cartesian3DPosition(2, 2, 2, frame),
                velocity=Cartesian3DVelocity(3, 3, 3, frame),
                frame=frame,
            ),
        ]
        trajectory = Trajectory.from_list_of_cartesian_state(states)
        np.testing.assert_array_equal(trajectory.time.et, [0, 1, 2])
        np.testing.assert_array_equal(
            trajectory.data[0], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        )
        np.testing.assert_array_equal(
            trajectory.data[1], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        )
        self.assertEqual(trajectory.frame, frame)

        # Test for mismatched frames
        states[1] = CartesianState(
            time=AbsoluteDate(1),
            position=Cartesian3DPosition(1, 1, 1, ReferenceFrame("ITRF")),
            velocity=Cartesian3DVelocity(2, 2, 2, ReferenceFrame("ITRF")),
            frame=ReferenceFrame("ITRF"),
        )
        with self.assertRaises(ValueError):
            Trajectory.from_list_of_cartesian_state(states)


class TestPositionSeries:
    """Unit tests for the PositionSeries class."""

    def test_initialization(self):
        """Test initialization of PositionSeries."""
        time = AbsoluteDateArray(np.array([0.0, 1.0]))
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frame = ReferenceFrame.ICRF_EC
        ps = PositionSeries(time, data, frame)
        assert np.array_equal(ps.data[0], data)
        assert ps.frame == frame

    def test_resample(self):
        """Test resampling of PositionSeries."""
        time = AbsoluteDateArray(np.array([0.0, 1.0]))
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frame = ReferenceFrame.ICRF_EC
        ps = PositionSeries(time, data, frame)
        new_time = np.array([0.5])
        resampled_ps = ps.resample(new_time)
        assert resampled_ps.data[0].shape == (1, 3)

    def test_remove_gaps(self):
        """Test removing gaps (NaN values) from PositionSeries."""
        time = AbsoluteDateArray(np.array([0.0, 1.0, 2.0]))
        data = np.array(
            [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]]
        )
        frame = ReferenceFrame.ICRF_EC
        ps = PositionSeries(time, data, frame)
        gapless_ps = ps.remove_gaps()
        assert gapless_ps.data[0].shape == (2, 3)

    def test_to_frame(self):
        """Test frame conversion for PositionSeries."""
        time = AbsoluteDateArray(np.array([0.0, 1.0]))
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
                np.array([0.0, 1.0])
            )  # Mock time attribute
        ps = PositionSeries.from_list_of_cartesian_position(positions)
        assert ps.data[0].shape == (2, 3)
        assert ps.frame == ReferenceFrame.ICRF_EC

    def test_arithmetic_operations(self):
        """Test arithmetic operations between PositionSeries."""
        time = AbsoluteDateArray(np.array([0.0, 1.0, 2.0]))
        data1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        data2 = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5]])
        frame = ReferenceFrame.ICRF_EC

        ps1 = PositionSeries(time, data1, frame)
        ps2 = PositionSeries(time, data2, frame)

        # Test addition
        added = ps1 + ps2
        np.testing.assert_allclose(added.data[0], data1 + data2)

        # Test subtraction
        subtracted = ps1 - ps2
        np.testing.assert_allclose(subtracted.data[0], data1 - data2)

        # Test multiplication
        multiplied = ps1 * 2
        np.testing.assert_allclose(multiplied.data[0], data1 * 2)

        # Test division
        divided = ps1 / 2
        np.testing.assert_allclose(divided.data[0], data1 / 2)

        # Test scalar addition
        scalar_added = ps1 + 1
        np.testing.assert_allclose(scalar_added.data[0], data1 + 1)

        # Test scalar subtraction
        scalar_subtracted = ps1 - 1
        np.testing.assert_allclose(scalar_subtracted.data[0], data1 - 1)


if __name__ == "__main__":
    unittest.main()

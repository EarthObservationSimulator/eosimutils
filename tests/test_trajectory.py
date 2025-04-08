"""Unit tests for eosimutils.trajectory module."""

import unittest
import numpy as np
from eosimutils.trajectory import Trajectory
from eosimutils.time import AbsoluteDates
from eosimutils.base import ReferenceFrame


class TestTrajectory(unittest.TestCase):
    """Unit tests for the Trajectory class."""

    def setUp(self):
        self.time = AbsoluteDates(np.array([0, 1, 2]))
        self.positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.velocities = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        self.frame = ReferenceFrame("GCRF")
        self.trajectory = Trajectory(self.time, [self.positions, self.velocities], self.frame)

    def test_resample(self):
        """Test resampling of trajectory data."""
        new_time = np.array([0.5, 1.5])
        resampled = self.trajectory.resample(new_time)
        self.assertEqual(resampled.time.et.tolist(), new_time.tolist())
        np.testing.assert_allclose(resampled.data[0], [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
        np.testing.assert_allclose(resampled.data[1], [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])

    def test_remove_gaps(self):
        """Test removal of gaps in trajectory data."""
        positions_with_gaps = np.array([[np.nan, np.nan, np.nan], [1, 1, 1],
                                        [np.nan, np.nan, np.nan]])
        velocities_with_gaps = np.array([[np.nan, np.nan, np.nan], [1, 1, 1],
                                         [np.nan, np.nan, np.nan]])
        trajectory_with_gaps = Trajectory(self.time, [positions_with_gaps, velocities_with_gaps],
                                          self.frame)
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
        other_trajectory = Trajectory(self.time, [other_positions, other_velocities], self.frame)

        added = self.trajectory + other_trajectory
        np.testing.assert_allclose(added.data[0], self.positions + other_positions)
        np.testing.assert_allclose(added.data[1], self.velocities + other_velocities)

        subtracted = self.trajectory - other_trajectory
        np.testing.assert_allclose(subtracted.data[0], self.positions - other_positions)
        np.testing.assert_allclose(subtracted.data[1], self.velocities - other_velocities)

    def test_constant_position(self):
        """Test creation of a constant position trajectory."""
        position = np.array([1, 1, 1])
        constant_traj = Trajectory.constant_position(0, 1, position, self.frame)
        np.testing.assert_allclose(constant_traj.data[0], [[1, 1, 1], [1, 1, 1]])
        np.testing.assert_allclose(constant_traj.data[1], [[0, 0, 0], [0, 0, 0]])

    def test_constant_velocity(self):
        """Test creation of a constant velocity trajectory."""
        initial_position = np.array([0, 0, 0])
        velocity = np.array([1, 1, 1])
        constant_vel_traj = Trajectory.constant_velocity(0, 1, velocity, initial_position,
                                                         self.frame)
        np.testing.assert_allclose(constant_vel_traj.data[0], [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_allclose(constant_vel_traj.data[1], [[1, 1, 1], [1, 1, 1]])

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of Trajectory."""
        serialized = self.trajectory.to_dict()
        deserialized = Trajectory.from_dict(serialized)
        np.testing.assert_allclose(deserialized.time.et, self.trajectory.time.et,
                                   rtol=1e-4,atol=1e-4)
        for d1, d2 in zip(deserialized.data, self.trajectory.data):
            np.testing.assert_allclose(d1, d2)
        self.assertEqual(deserialized.frame, self.trajectory.frame)


if __name__ == "__main__":
    unittest.main()

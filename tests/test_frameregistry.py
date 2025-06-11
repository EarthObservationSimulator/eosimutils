"""Unit tests for eosimutils.frame_registry module."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

from eosimutils.frame_registry import FrameRegistry
from eosimutils.orientation import ConstantOrientation, OrientationSeries
from eosimutils.base import ReferenceFrame
from eosimutils.time import AbsoluteDate, AbsoluteDateArray

ReferenceFrame.add("A")
ReferenceFrame.add("B")
ReferenceFrame.add("C")
ReferenceFrame.add("D")
ReferenceFrame.add("E")
ReferenceFrame.add("F")


class TestFrameRegistry(unittest.TestCase):
    """Tests for graph-based frame lookups in FrameRegistry."""

    def setUp(self):
        self.registry = FrameRegistry()

        # Define 90 deg rotations about Z for edges
        rot90 = R.from_euler("xyz", [0, 0, 90], degrees=True)

        # Create cycle in graph: A->B->C->D->A
        self.ab = ConstantOrientation(
            rot90, ReferenceFrame.get("A"), ReferenceFrame.get("B")
        )
        self.registry.add_transform(self.ab)
        self.bc = ConstantOrientation(
            rot90, ReferenceFrame.get("B"), ReferenceFrame.get("C")
        )
        self.registry.add_transform(self.bc)
        self.cd = ConstantOrientation(
            rot90, ReferenceFrame.get("C"), ReferenceFrame.get("D")
        )
        self.registry.add_transform(self.cd)
        self.da = ConstantOrientation(
            rot90, ReferenceFrame.get("D"), ReferenceFrame.get("A")
        )
        self.registry.add_transform(self.da)

        # Create a direct edge from A->C
        # which gives a shorter path from A->C than A->B->C
        self.ad = ConstantOrientation(
            rot90, ReferenceFrame.get("A"), ReferenceFrame.get("C")
        )
        self.registry.add_transform(self.ad)

        self.ae = ConstantOrientation(
            rot90, ReferenceFrame.get("A"), ReferenceFrame.get("E")
        )
        self.registry.add_transform(self.ae)

    def test_direct_transform(self):
        """A single edge A->B should produce a 90 deg rotation about Z."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("B"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_inverse_transform(self):
        """Requesting B->A uses the auto-registered inverse, yielding −90 deg about Z."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.get("B"), ReferenceFrame.get("A"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, -90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_two_step_transform(self):
        """Two‐step path B→D (B→A→D) yields a -180 deg rotation about Z."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.get("B"), ReferenceFrame.get("D"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, -180], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_missing_path_raises(self):
        """Asking for a non-connected path (A->F) raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_transform(
                ReferenceFrame.get("A"),
                ReferenceFrame.get("F"),
                AbsoluteDate(0.0),
            )

    def test_closest_path(self):
        """Asking for a transform from (A->C) should return shortest (direct) path.
        which is the 90 deg rotation."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("C"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_constant_transform_with_none_time(self):
        """Test that get_transform works with t=None for constant orientations."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("B"), None
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_non_constant_transform_with_none_time(self):
        """Test that get_transform raises KeyError for non-constant orientations when t=None."""
        # Add a non-constant orientation to the registry
        time = AbsoluteDateArray(np.array([0.0, 0.5, 1.0]))
        rotations = R.from_euler(
            "xyz", [[0, 0, 0], [0, 0, 45], [0, 0, 90]], degrees=True
        )
        non_constant_orientation = OrientationSeries(
            time,
            rotations,
            ReferenceFrame.get("B"),
            ReferenceFrame.get("F"),
        )
        self.registry.add_transform(non_constant_orientation)

        with self.assertRaises(KeyError):
            self.registry.get_transform(
                ReferenceFrame.get("A"), ReferenceFrame.get("F"), None
            )

    def test_no_path_with_none_time(self):
        """Test that get_transform raises KeyError when no path exists with t=None."""
        with self.assertRaises(KeyError):
            self.registry.get_transform(
                ReferenceFrame.get("A"), ReferenceFrame.get("F"), None
            )


if __name__ == "__main__":
    unittest.main()

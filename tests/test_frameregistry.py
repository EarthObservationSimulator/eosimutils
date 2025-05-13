"""Unit tests for eosimutils.frame_registry module."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

from eosimutils.frame_registry import FrameRegistry
from eosimutils.orientation import ConstantOrientation
from eosimutils.frames import ReferenceFrame
from eosimutils.time import AbsoluteDate

# Define frames
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
        self.ab = ConstantOrientation(rot90, ReferenceFrame.A, ReferenceFrame.B)
        self.registry.add_transform(self.ab)
        self.bc = ConstantOrientation(rot90, ReferenceFrame.B, ReferenceFrame.C)
        self.registry.add_transform(self.bc)
        self.cd = ConstantOrientation(rot90, ReferenceFrame.C, ReferenceFrame.D)
        self.registry.add_transform(self.cd)
        self.da = ConstantOrientation(rot90, ReferenceFrame.D, ReferenceFrame.A)
        self.registry.add_transform(self.da)

        # Create a direct edge from A->C
        # which gives a shorter path from A->C than A->B->C
        self.ad = ConstantOrientation(rot90, ReferenceFrame.A, ReferenceFrame.C)
        self.registry.add_transform(self.ad)

        self.ae = ConstantOrientation(rot90, ReferenceFrame.A, ReferenceFrame.E)
        self.registry.add_transform(self.ae)

    def test_direct_transform(self):
        """A single edge A->B should produce a 90 deg rotation about Z."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.A, ReferenceFrame.B, AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_inverse_transform(self):
        """Requesting B->A uses the auto-registered inverse, yielding −90 deg about Z."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.B, ReferenceFrame.A, AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, -90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_two_step_transform(self):
        """Two‐step path B→D (B→A→D) yields a -180 deg rotation about Z."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.B, ReferenceFrame.D, AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, -180], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_missing_path_raises(self):
        """Asking for a non-connected path (A->F) raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_transform(
                ReferenceFrame.A, ReferenceFrame.F, AbsoluteDate(0.0)
            )

    def test_closest_path(self):
        """Asking for a transform from (A->C) should return shortest (direct) path.
        which is the 90 deg rotation."""
        rot, omega = self.registry.get_transform(
            ReferenceFrame.A, ReferenceFrame.C, AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))


if __name__ == "__main__":
    unittest.main()

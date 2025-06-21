"""Unit tests for eosimutils.frame_registry module."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

from eosimutils.framegraph import FrameGraph
from eosimutils.orientation import ConstantOrientation, OrientationSeries
from eosimutils.base import ReferenceFrame
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.state import Cartesian3DPosition

ReferenceFrame.add("A")
ReferenceFrame.add("B")
ReferenceFrame.add("C")
ReferenceFrame.add("D")
ReferenceFrame.add("E")
ReferenceFrame.add("F")


class TestFrameGraph(unittest.TestCase):
    """Tests for graph-based frame lookups in FrameGraph."""

    def setUp(self):
        self.registry = FrameGraph()

        # Define 90 deg rotations about Z for edges
        rot90 = R.from_euler("xyz", [0, 0, 90], degrees=True)

        # Create cycle in graph: A->B->C->D->A
        self.ab = ConstantOrientation(
            rot90, ReferenceFrame.get("A"), ReferenceFrame.get("B")
        )
        self.registry.add_orientation_transform(self.ab)
        self.bc = ConstantOrientation(
            rot90, ReferenceFrame.get("B"), ReferenceFrame.get("C")
        )
        self.registry.add_orientation_transform(self.bc)
        self.cd = ConstantOrientation(
            rot90, ReferenceFrame.get("C"), ReferenceFrame.get("D")
        )
        self.registry.add_orientation_transform(self.cd)
        self.da = ConstantOrientation(
            rot90, ReferenceFrame.get("D"), ReferenceFrame.get("A")
        )
        self.registry.add_orientation_transform(self.da)

        # Create a direct edge from A->C
        self.ac = ConstantOrientation(
            rot90, ReferenceFrame.get("A"), ReferenceFrame.get("C")
        )
        self.registry.add_orientation_transform(self.ac)

        self.ae = ConstantOrientation(
            rot90, ReferenceFrame.get("A"), ReferenceFrame.get("E")
        )
        self.registry.add_orientation_transform(self.ae)

        # Position transforms for the same edges as orientation transforms
        self.p_ab = Cartesian3DPosition(1, 0, 0, ReferenceFrame.get("A"))
        self.registry.add_pos_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("B"), self.p_ab
        )
        self.p_bc = Cartesian3DPosition(2, 0, 0, ReferenceFrame.get("B"))
        self.registry.add_pos_transform(
            ReferenceFrame.get("B"), ReferenceFrame.get("C"), self.p_bc
        )
        self.p_cd = Cartesian3DPosition(3, 0, 0, ReferenceFrame.get("C"))
        self.registry.add_pos_transform(
            ReferenceFrame.get("C"), ReferenceFrame.get("D"), self.p_cd
        )
        self.p_da = Cartesian3DPosition(4, 0, 0, ReferenceFrame.get("D"))
        self.registry.add_pos_transform(
            ReferenceFrame.get("D"), ReferenceFrame.get("A"), self.p_da
        )
        self.p_ac = Cartesian3DPosition(5, 0, 0, ReferenceFrame.get("A"))
        self.registry.add_pos_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("C"), self.p_ac
        )
        self.p_ae = Cartesian3DPosition(6, 0, 0, ReferenceFrame.get("A"))
        self.registry.add_pos_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("E"), self.p_ae
        )

        # Round-trip the registry to ensure it can be serialized and deserialized
        # Comment these lines to test without serialization
        registry_dict = self.registry.to_dict()
        self.registry = FrameGraph.from_dict(registry_dict)

    def test_direct_transform(self):
        """A single edge A->B should produce a 90 deg rotation about Z."""
        rot, omega = self.registry.get_orientation_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("B"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_inverse_transform(self):
        """Requesting B->A uses the auto-registered inverse, yielding −90 deg about Z."""
        rot, omega = self.registry.get_orientation_transform(
            ReferenceFrame.get("B"), ReferenceFrame.get("A"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, -90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_two_step_transform(self):
        """Two‐step path B→D (B→A→D) yields a -180 deg rotation about Z."""
        rot, omega = self.registry.get_orientation_transform(
            ReferenceFrame.get("B"), ReferenceFrame.get("D"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, -180], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_missing_path_raises(self):
        """Asking for a non-connected path (A->F) raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_orientation_transform(
                ReferenceFrame.get("A"),
                ReferenceFrame.get("F"),
                AbsoluteDate(0.0),
            )

    def test_closest_path(self):
        """Asking for a transform from (A->C) should return shortest (direct) path.
        which is the 90 deg rotation."""
        rot, omega = self.registry.get_orientation_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("C"), AbsoluteDate(0.0)
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_constant_transform_with_none_time(self):
        """Test that get_orientation_transform works with t=None for constant orientations."""
        rot, omega = self.registry.get_orientation_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("B"), None
        )
        np.testing.assert_allclose(
            rot.as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-8
        )
        np.testing.assert_array_equal(omega, np.zeros(3))

    def test_non_constant_transform_with_none_time(self):
        """Test that get_orientation_transform raises KeyError for non-constant orientations when t=None."""
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
        self.registry.add_orientation_transform(non_constant_orientation)

        with self.assertRaises(KeyError):
            self.registry.get_orientation_transform(
                ReferenceFrame.get("A"), ReferenceFrame.get("F"), None
            )

    def test_no_path_with_none_time(self):
        """Test that get_orientation_transform raises KeyError when no path exists with t=None."""
        with self.assertRaises(KeyError):
            self.registry.get_orientation_transform(
                ReferenceFrame.get("A"), ReferenceFrame.get("F"), None
            )

    # Position transform tests

    def test_direct_pos_transform(self):
        """A single edge A->B should produce the direct position vector."""
        offset = self.registry.get_pos_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("B"), None
        )
        np.testing.assert_array_equal(offset, [1, 0, 0])

    def test_inverse_pos_transform(self):
        """Requesting B->A uses the auto-registered inverse, yielding the correct vector
        expressed in the B frame."""
        # Use the rotation from self.ab (A->B)
        expected = -self.ab.rotation.apply([1, 0, 0])
        offset = self.registry.get_pos_transform(
            ReferenceFrame.get("B"), ReferenceFrame.get("A"), None
        )
        np.testing.assert_allclose(offset, expected)

    def test_two_step_pos_transform(self):
        """Two-step path B→D (B→A→D) yields the correct vector in B-frame coordinates."""
        # rot_bc is the transformation from C frame to B frame.

        rot_a_to_b = self.ab.rotation
        rot_d_to_a = self.da.rotation
        rot_d_to_b = rot_a_to_b * rot_d_to_a
        # Position vector from B to A in B coordinates
        p_ba = -rot_a_to_b.apply(self.p_ab.to_numpy())
        # Position from A to D in B coordinates
        p_ad = -rot_d_to_b.apply(self.p_da.to_numpy())
        # Position from B to D in B coordinates
        p_bd = p_ba + p_ad

        offset = self.registry.get_pos_transform(
            ReferenceFrame.get("B"), ReferenceFrame.get("D"), None
        )
        np.testing.assert_allclose(offset, p_bd)

    def test_missing_pos_path_raises(self):
        """Asking for a non-connected path (A->F) raises KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_pos_transform(
                ReferenceFrame.get("A"),
                ReferenceFrame.get("F"),
                None,
            )

    def test_closest_pos_path(self):
        """Asking for a position transform from (A->C) should return shortest (direct) path."""
        offset = self.registry.get_pos_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("C"), None
        )
        np.testing.assert_allclose(offset, [5, 0, 0])

    def test_pos_transform_with_none_time(self):
        """Test that get_pos_transform works with t=None for constant positions."""
        offset = self.registry.get_pos_transform(
            ReferenceFrame.get("A"), ReferenceFrame.get("B"), None
        )
        np.testing.assert_allclose(offset, [1, 0, 0])

    def test_no_pos_path_with_none_time(self):
        """Test that get_pos_transform raises KeyError when no path exists with t=None."""
        with self.assertRaises(KeyError):
            self.registry.get_pos_transform(
                ReferenceFrame.get("A"), ReferenceFrame.get("F"), None
            )


if __name__ == "__main__":
    unittest.main()

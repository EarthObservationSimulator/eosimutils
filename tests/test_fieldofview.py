"""Unit tests for the fieldofview module."""

import unittest
from eosimutils.fieldofview import (
    FieldOfViewFactory,
    FieldOfViewType,
    CircularFieldOfView,
    RectangularFieldOfView,
    PolygonFieldOfView,
)


class TestFieldOfViewFactory(unittest.TestCase):
    """Unit tests for the FieldOfViewFactory class."""

    def setUp(self):
        """Set up test data for FieldOfViewFactory."""
        self.factory = FieldOfViewFactory()
        self.circular_fov_specs = {
            "fov_type": FieldOfViewType.CIRCULAR.value,
            "diameter": 60.0,
            "frame": "ICRF_EC",
            "boresight": [0.0, 0.0, 1.0],
        }

    def test_get_fov_circular(self):
        """Test retrieving a CircularFieldOfView object."""
        fov = self.factory.get_fov(self.circular_fov_specs)
        self.assertIsInstance(fov, CircularFieldOfView)
        self.assertEqual(fov.diameter, self.circular_fov_specs["diameter"])
        self.assertEqual(fov.frame, self.circular_fov_specs["frame"])
        self.assertEqual(
            fov.boresight.tolist(), self.circular_fov_specs["boresight"]
        )

    def test_get_fov_missing_type(self):
        """Test error handling for a missing fov_type key."""
        specs = {"diameter": 60.0}
        with self.assertRaises(KeyError) as context:
            self.factory.get_fov(specs)
        self.assertIn(
            'FOV type key "fov_type" not found in specifications dictionary.',
            str(context.exception),
        )

    def test_get_fov_invalid_type(self):
        """Test error handling for an invalid fov_type."""
        specs = {"fov_type": "INVALID_TYPE", "diameter": 60.0}
        with self.assertRaises(ValueError) as context:
            self.factory.get_fov(specs)
        self.assertIn(
            'FOV type "INVALID_TYPE" is not registered.', str(context.exception)
        )

    def test_register_fov(self):
        """Test registering a new FOV type."""

        class DummyFOV:
            @classmethod
            def from_dict(cls, specs):
                self.specs = specs
                return cls()

        self.factory.register_fov("DUMMY_FOV", DummyFOV)
        specs = {"fov_type": "DUMMY_FOV"}
        fov = self.factory.get_fov(specs)
        self.assertIsInstance(fov, DummyFOV)


class TestCircularFieldOfView(unittest.TestCase):
    """Unit tests for the CircularFieldOfView class."""

    def setUp(self):
        """Set up test data for CircularFieldOfView."""
        self.diameter = 60.0
        self.frame = "ICRF_EC"
        self.boresight = [0.0, 0.0, 1.0]
        self.circular_fov = CircularFieldOfView(
            diameter=self.diameter, frame=self.frame, boresight=self.boresight
        )

    def test_initialization(self):
        """Test initialization of CircularFieldOfView."""
        self.assertEqual(self.circular_fov.diameter, self.diameter)
        self.assertEqual(self.circular_fov.frame, self.frame)
        self.assertTrue((self.circular_fov.boresight == self.boresight).all())

    def test_from_dict(self):
        """Test creating CircularFieldOfView from a dictionary."""
        specs = {
            "diameter": self.diameter,
            "frame": self.frame,
            "boresight": self.boresight,
        }
        circular_fov = CircularFieldOfView.from_dict(specs)
        self.assertEqual(circular_fov.diameter, self.diameter)
        self.assertEqual(circular_fov.frame, self.frame)
        self.assertTrue((circular_fov.boresight == self.boresight).all())

    def test_default_boresight(self):
        """Test default boresight vector."""
        specs = {
            "diameter": self.diameter,
            "frame": self.frame,
        }
        circular_fov = CircularFieldOfView.from_dict(specs)
        self.assertTrue((circular_fov.boresight == [0.0, 0.0, 1.0]).all())

    def test_missing_boresight(self):
        """Test creating CircularFieldOfView from a dictionary without the boresight key."""
        specs = {
            "diameter": self.diameter,
            "frame": self.frame,
        }
        circular_fov = CircularFieldOfView.from_dict(specs)
        self.assertTrue((circular_fov.boresight == [0.0, 0.0, 1.0]).all())

    def test_to_dict(self):
        """Test converting CircularFieldOfView to a dictionary."""
        fov_dict = self.circular_fov.to_dict()
        self.assertEqual(fov_dict["diameter"], self.diameter)
        self.assertEqual(fov_dict["frame"], self.frame)
        self.assertEqual(fov_dict["boresight"], self.boresight)

    def test_diameter_range(self):
        """Test that diameter must be between 0 and 180 degrees."""
        specs = {
            "diameter": 200.0,  # Invalid diameter
            "frame": self.frame,
            "boresight": self.boresight,
        }
        with self.assertRaises(ValueError) as context:
            CircularFieldOfView.from_dict(specs)
        self.assertIn(
            "diameter must be between 0 and 180 degrees.",
            str(context.exception),
        )

        specs = {
            "diameter": -1.0,  # Invalid diameter
            "frame": self.frame,
            "boresight": self.boresight,
        }
        with self.assertRaises(ValueError) as context:
            CircularFieldOfView.from_dict(specs)
        self.assertIn(
            "diameter must be between 0 and 180 degrees.",
            str(context.exception),
        )


class TestRectangularFieldOfView(unittest.TestCase):
    """Unit tests for the RectangularFieldOfView class."""

    def setUp(self):
        """Set up test data for RectangularFieldOfView."""
        self.frame = "ICRF_EC"
        self.boresight = [0.0, 0.0, 1.0]
        self.ref_vector = [1.0, 0.0, 0.0]
        self.ref_angle = 45.0
        self.cross_angle = 30.0
        self.rectangular_fov = RectangularFieldOfView(
            frame=self.frame,
            boresight=self.boresight,
            ref_vector=self.ref_vector,
            ref_angle=self.ref_angle,
            cross_angle=self.cross_angle,
        )

    def test_initialization(self):
        """Test initialization of RectangularFieldOfView."""
        self.assertEqual(self.rectangular_fov.frame, self.frame)
        self.assertTrue(
            (self.rectangular_fov.boresight == self.boresight).all()
        )
        self.assertTrue(
            (self.rectangular_fov.ref_vector == self.ref_vector).all()
        )
        self.assertEqual(self.rectangular_fov.ref_angle, self.ref_angle)
        self.assertEqual(self.rectangular_fov.cross_angle, self.cross_angle)

    def test_from_dict(self):
        """Test creating RectangularFieldOfView from a dictionary."""
        specs = {
            "frame": self.frame,
            "boresight": self.boresight,
            "ref_vector": self.ref_vector,
            "ref_angle": self.ref_angle,
            "cross_angle": self.cross_angle,
        }
        rectangular_fov = RectangularFieldOfView.from_dict(specs)
        self.assertEqual(rectangular_fov.frame, self.frame)
        self.assertTrue((rectangular_fov.boresight == self.boresight).all())
        self.assertTrue((rectangular_fov.ref_vector == self.ref_vector).all())
        self.assertEqual(rectangular_fov.ref_angle, self.ref_angle)
        self.assertEqual(rectangular_fov.cross_angle, self.cross_angle)

    def test_missing_boresight(self):
        """Test creating RectangularFieldOfView from a dictionary without the boresight key."""
        specs = {
            "frame": self.frame,
            "ref_vector": self.ref_vector,
            "ref_angle": self.ref_angle,
            "cross_angle": self.cross_angle,
        }
        rectangular_fov = RectangularFieldOfView.from_dict(specs)
        self.assertTrue((rectangular_fov.boresight == [0.0, 0.0, 1.0]).all())

    def test_to_dict(self):
        """Test converting RectangularFieldOfView to a dictionary."""
        fov_dict = self.rectangular_fov.to_dict()
        self.assertEqual(fov_dict["frame"], self.frame)
        self.assertEqual(fov_dict["boresight"], self.boresight)
        self.assertEqual(fov_dict["ref_vector"], self.ref_vector)
        self.assertEqual(fov_dict["ref_angle"], self.ref_angle)
        self.assertEqual(fov_dict["cross_angle"], self.cross_angle)

    def test_angle_ranges(self):
        """Test that ref_angle and cross_angle must be between 0 and 90 degrees."""
        specs_invalid_ref_angle = {
            "frame": self.frame,
            "boresight": self.boresight,
            "ref_vector": self.ref_vector,
            "ref_angle": 100.0,  # Invalid ref_angle
            "cross_angle": self.cross_angle,
        }
        with self.assertRaises(ValueError) as context:
            RectangularFieldOfView.from_dict(specs_invalid_ref_angle)
        self.assertIn(
            "ref_angle must be between 0 and 90 degrees.",
            str(context.exception),
        )

        specs_invalid_cross_angle = {
            "frame": self.frame,
            "boresight": self.boresight,
            "ref_vector": self.ref_vector,
            "ref_angle": self.ref_angle,
            "cross_angle": -10.0,  # Invalid cross_angle
        }
        with self.assertRaises(ValueError) as context:
            RectangularFieldOfView.from_dict(specs_invalid_cross_angle)
        self.assertIn(
            "cross_angle must be between 0 and 90 degrees.",
            str(context.exception),
        )


class TestPolygonFieldOfView(unittest.TestCase):
    """Unit tests for the PolygonFieldOfView class."""

    def setUp(self):
        """Set up test data for PolygonFieldOfView."""
        self.frame = "ICRF_EC"
        self.boresight = [0.0, 0.0, 1.0]
        self.boundary_corners = [
            [0.5, 0.5, 0.707],
            [0.1, 0.2, 0.979],
            [0.3, 0.4, 0.866],
            [0.6, 0.0, 0.8],
        ]
        self.polygon_fov = PolygonFieldOfView(
            frame=self.frame,
            boresight=self.boresight,
            boundary_corners=self.boundary_corners,
        )

    def test_initialization(self):
        """Test initialization of PolygonFieldOfView."""
        self.assertEqual(self.polygon_fov.frame, self.frame)
        self.assertTrue((self.polygon_fov.boresight == self.boresight).all())
        for corner, expected_corner in zip(
            self.polygon_fov.boundary_corners, self.boundary_corners
        ):
            self.assertTrue((corner == expected_corner).all())

    def test_from_dict(self):
        """Test creating PolygonFieldOfView from a dictionary."""
        specs = {
            "frame": self.frame,
            "boresight": self.boresight,
            "boundary_corners": self.boundary_corners,
        }
        polygon_fov = PolygonFieldOfView.from_dict(specs)
        self.assertEqual(polygon_fov.frame, self.frame)
        self.assertTrue((polygon_fov.boresight == self.boresight).all())
        for corner, expected_corner in zip(
            polygon_fov.boundary_corners, self.boundary_corners
        ):
            self.assertTrue((corner == expected_corner).all())

    def test_to_dict(self):
        """Test converting PolygonFieldOfView to a dictionary."""
        fov_dict = self.polygon_fov.to_dict()
        self.assertEqual(fov_dict["frame"], self.frame)
        self.assertEqual(fov_dict["boresight"], self.boresight)
        self.assertEqual(fov_dict["boundary_corners"], self.boundary_corners)

    def test_missing_boresight(self):
        """Test creating PolygonFieldOfView from a dictionary without the boresight key."""
        specs = {
            "frame": self.frame,
            "boundary_corners": self.boundary_corners,
        }
        polygon_fov = PolygonFieldOfView.from_dict(specs)
        self.assertTrue((polygon_fov.boresight == [0.0, 0.0, 1.0]).all())

    def test_invalid_boundary_corners_count(self):
        """Test that an error is raised if fewer than 3 boundary corners are provided."""
        invalid_corners = [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
        ]  # Only 2 corners provided
        specs = {
            "frame": self.frame,
            "boresight": self.boresight,
            "boundary_corners": invalid_corners,
        }
        with self.assertRaises(ValueError) as context:
            PolygonFieldOfView.from_dict(specs)
        self.assertIn(
            "At least 3 vectors must be defined in boundary_corners.",
            str(context.exception),
        )

    def test_invalid_boundary_corners_hemisphere(self):
        """Test that an error is raised if a boundary corner is not in the same
        hemisphere as the boresight."""
        invalid_corners = [
            [1.0, 0.0, -0.5],  # This vector is in the -Z hemisphere
            [0.0, 1.0, 0.5],
            [-1.0, 0.0, 0.5],
        ]
        specs = {
            "frame": self.frame,
            "boresight": self.boresight,
            "boundary_corners": invalid_corners,
        }
        with self.assertRaises(ValueError) as context:
            PolygonFieldOfView.from_dict(specs)
        self.assertIn(
            "All boundary_corners must be in the same hemisphere as the boresight vector.",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()

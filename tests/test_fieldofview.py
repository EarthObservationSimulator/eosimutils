"""Unit tests for the fieldofview module."""

import unittest
from eosimutils.fieldofview import (
    FieldOfViewFactory,
    FieldOfViewType,
    CircularFieldOfView,
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
        self.assertEqual(fov.boresight.tolist(), self.circular_fov_specs["boresight"])

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
        self.assertEqual(self.circular_fov.frame.value, self.frame)
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
        self.assertEqual(circular_fov.frame.value, self.frame)
        self.assertTrue((circular_fov.boresight == self.boresight).all())
    
    def test_default_boresight(self):
        """Test default boresight vector."""
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


if __name__ == "__main__":
    unittest.main()

"""Unit tests for eosimutils.orientation.SpiceOrientation class."""

import unittest
import numpy as np
import random

from eosimutils.base import ReferenceFrame
from eosimutils.state import (
    Cartesian3DPosition,
    Cartesian3DVelocity,
    CartesianState,
)
from eosimutils.time import AbsoluteDate
from eosimutils.orientation import SpiceOrientation

from testutils import (
    validate_transform_position_with_astropy,
    validate_transform_state_with_astropy,
)

# Shared constants for the tests
# Example position in inertial frame
_TEST_POSITION_I = Cartesian3DPosition.from_dict(
    {
        "x": random.uniform(-10000, 10000),
        "y": random.uniform(-10000, 10000),
        "z": random.uniform(-10000, 10000),
        "frame": "ICRF_EC",
    }
)
_TEST_VELOCITY_I = Cartesian3DVelocity.from_dict(
    {
        "vx": random.uniform(-10, 10),
        "vy": random.uniform(-10, 10),
        "vz": random.uniform(-10, 10),
        "frame": "ICRF_EC",
    }
)

# Example time
_TEST_YEAR = random.randint(2000, 2024)
_TEST_MONTH = random.randint(1, 12)
_TEST_DAY = random.randint(1, 28)  # To avoid invalid dates
_TEST_HOUR = random.randint(0, 23)
_TEST_MINUTE = random.randint(0, 59)
_TEST_SECOND = random.randint(0, 59)
_TEST_TIME = AbsoluteDate.from_dict(
    {
        "time_format": "Gregorian_Date",
        "calendar_date": (
            f"{_TEST_YEAR:04d}-"
            f"{_TEST_MONTH:02d}-"
            f"{_TEST_DAY:02d}T"
            f"{_TEST_HOUR:02d}:"
            f"{_TEST_MINUTE:02d}:"
            f"{_TEST_SECOND:02d}"
        ),
        "time_scale": "UTC",
    }
)

# Example state in inertial frame
_TEST_STATE_I = CartesianState(
    _TEST_TIME,
    _TEST_POSITION_I,
    _TEST_VELOCITY_I,
    ReferenceFrame.get("ICRF_EC"),
)

# Example position in Earth-fixed frame
_TEST_POSITION_EF = Cartesian3DPosition.from_dict(
    {
        "x": random.uniform(-10000, 10000),
        "y": random.uniform(-10000, 10000),
        "z": random.uniform(-10000, 10000),
        "frame": "ITRF",
    }
)
_TEST_VELOCITY_EF = Cartesian3DVelocity.from_dict(
    {
        "vx": random.uniform(-10, 10),
        "vy": random.uniform(-10, 10),
        "vz": random.uniform(-10, 10),
        "frame": "ITRF",
    }
)
# Example state in EF frame
_TEST_STATE_EF = CartesianState(
    _TEST_TIME, _TEST_POSITION_EF, _TEST_VELOCITY_EF, ReferenceFrame.get("ITRF")
)


class TestSpiceOrientation(unittest.TestCase):
    """Unit tests for the SpiceOrientation class."""

    def setUp(self):
        self.from_frame = ReferenceFrame.get("ICRF_EC")
        self.to_frame = ReferenceFrame.get("ITRF")
        self.orientation = SpiceOrientation(self.from_frame, self.to_frame)
        self.time = _TEST_TIME

    def test_construction(self):
        self.assertEqual(self.orientation.from_frame, self.from_frame)
        self.assertEqual(self.orientation.to_frame, self.to_frame)

    def test_to_dict_and_from_dict(self):
        d = self.orientation.to_dict()
        self.assertEqual(d["orientation_type"], "spice")
        self.assertEqual(d["from"], self.from_frame.to_string())
        self.assertEqual(d["to"], self.to_frame.to_string())
        orientation2 = SpiceOrientation.from_dict(d)
        self.assertEqual(orientation2.from_frame, self.from_frame)
        self.assertEqual(orientation2.to_frame, self.to_frame)

    def test_inverse(self):
        inv = self.orientation.inverse()
        self.assertEqual(inv.from_frame, self.to_frame)
        self.assertEqual(inv.to_frame, self.from_frame)
        # Inverse of inverse should be original
        orig = inv.inverse()
        self.assertEqual(orig.from_frame, self.from_frame)
        self.assertEqual(orig.to_frame, self.to_frame)

    def test_at_method_type(self):
        # This test checks that at() returns a tuple of (Scipy_Rotation, np.ndarray)
        result = self.orientation.at(self.time)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestSpiceOrientationTransformState(unittest.TestCase):
    """Unit tests for the SpiceOrientation.transform_position function."""

    def test_no_transformation(self):
        """Test no transformation when from_frame and to_frame are the same."""
        test_orientation = SpiceOrientation(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
        )
        transform_state = test_orientation.transform_state
        transformed_state = transform_state(
            state=_TEST_STATE_I,
        )
        np.testing.assert_array_equal(
            transformed_state.position.to_list(),
            _TEST_STATE_I.position.to_list(),
        )
        np.testing.assert_array_equal(
            transformed_state.velocity.to_list(),
            _TEST_STATE_I.velocity.to_list(),
        )
        self.assertEqual(transformed_state.frame, ReferenceFrame.get("ICRF_EC"))

    def test_transform_icrfec_to_itrf(self):
        """Test transformation from ICRF_EC to ITRF."""
        test_orientation = SpiceOrientation(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
        )
        transform_state = test_orientation.transform_state
        transformed_state = transform_state(
            state=_TEST_STATE_I,
        )
        # Validate the transformed state
        self.assertEqual(transformed_state.frame, ReferenceFrame.get("ITRF"))
        self.assertEqual(len(transformed_state.position.to_list()), 3)
        self.assertEqual(len(transformed_state.velocity.to_list()), 3)
        self.assertTrue(
            all(
                isinstance(coord, float)
                for coord in transformed_state.position.to_list()
            )
        )
        self.assertTrue(
            all(
                isinstance(coord, float)
                for coord in transformed_state.velocity.to_list()
            )
        )

    def test_invalid_frame(self):
        """Test transformation with a non-matching frame of the input state vector."""
        ReferenceFrame.add("XYZ")
        test_orientation = SpiceOrientation(
            from_frame=ReferenceFrame.get("XYZ"),
            to_frame=ReferenceFrame.get("ITRF"),
        )
        transform_state = test_orientation.transform_state
        with self.assertRaises(ValueError):
            transform_state(
                state=_TEST_STATE_I,
            )

    def test_round_trip_transformation(self):
        """Test round-trip transformation from ICRF_EC to ITRF and back to ICRF_EC."""
        test_orientation_1 = SpiceOrientation(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
        )
        transform_state_1 = test_orientation_1.transform_state

        # Transform state from ICRF_EC to ITRF
        transformed_to_itrf = transform_state_1(
            state=_TEST_STATE_I,
        )

        test_orientation_2 = SpiceOrientation(
            from_frame=ReferenceFrame.get("ITRF"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
        )
        transform_state_2 = test_orientation_2.transform_state

        # Transform state back from ITRF to ICRF_EC
        transformed_back_to_icrf = transform_state_2(
            state=transformed_to_itrf,
        )

        # Compare the original state to the resulting state
        np.testing.assert_allclose(
            _TEST_STATE_I.position.to_numpy(),
            transformed_back_to_icrf.position.to_numpy(),
            atol=1e-6,
            err_msg="Round-trip transformation failed for position: ICRF_EC -> ITRF -> ICRF_EC",
        )
        np.testing.assert_allclose(
            _TEST_STATE_I.velocity.to_numpy(),
            transformed_back_to_icrf.velocity.to_numpy(),
            atol=1e-6,
            err_msg="Round-trip transformation failed for velocity: ICRF_EC -> ITRF -> ICRF_EC",
        )
        self.assertEqual(
            transformed_back_to_icrf.frame,
            ReferenceFrame.get("ICRF_EC"),
            "The resulting frame is not ICRF_EC after round-trip transformation.",
        )

    def test_transform_state_with_astropy(self):
        """Test transformation using astropy_transform for validation.
        Validates both position and velocity transformations."""
        # Validate the state transformation from ICRF_EC to ITRF
        is_valid = validate_transform_state_with_astropy(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
            state=_TEST_STATE_I,
        )
        self.assertTrue(
            is_valid, "Transform state validation failed for ICRF_EC to ITRF."
        )

        # Validate the state transformation from ITRF to ICRF_EC
        is_valid = validate_transform_state_with_astropy(
            from_frame=ReferenceFrame.get("ITRF"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
            state=_TEST_STATE_EF,
        )
        self.assertTrue(
            is_valid, "Transform state validation failed for ITRF to ICRF_EC."
        )


class TestSpiceOrientationTransformPosition(unittest.TestCase):
    """Test the SpiceOrientation.tranform_position function."""

    def test_no_transformation(self):
        """Test no transformation when from_frame and to_frame are the same."""
        test_orientation = SpiceOrientation(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
        )
        transform_position = test_orientation.transform_position
        transformed_position = transform_position(
            position=_TEST_POSITION_I,
            t=_TEST_TIME,
        )
        np.testing.assert_array_equal(
            transformed_position.to_list(), _TEST_POSITION_I.to_list()
        )
        self.assertEqual(
            transformed_position.frame, ReferenceFrame.get("ICRF_EC")
        )

    def test_transform_icrfec_to_itrf(self):
        """Test transformation from ICRF_EC to ITRF."""
        test_orientation = SpiceOrientation(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
        )
        transform_position = test_orientation.transform_position
        transformed_position = transform_position(
            position=_TEST_POSITION_I,
            t=_TEST_TIME,
        )
        # Validate the transformed position
        self.assertEqual(transformed_position.frame, ReferenceFrame.get("ITRF"))
        self.assertEqual(len(transformed_position.to_list()), 3)
        self.assertTrue(
            all(
                isinstance(coord, float)
                for coord in transformed_position.to_list()
            )
        )

    def test_invalid_frame(self):
        """Test transformation with invalid frames."""
        ReferenceFrame.add("ABC")
        test_orientation = SpiceOrientation(
            from_frame=ReferenceFrame.get("ABC"),
            to_frame=ReferenceFrame.get("ITRF"),
        )
        transform_position = test_orientation.transform_position
        with self.assertRaises(ValueError):
            transform_position(
                position=_TEST_POSITION_I,
                t=_TEST_TIME,
            )

    def test_round_trip_transformation(self):
        """Test round-trip transformation from ICRF_EC to ITRF and back to ICRF_EC."""
        test_orientation_1 = SpiceOrientation(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
        )
        transform_position_1 = test_orientation_1.transform_position

        # Transform position from ICRF_EC to ITRF
        transformed_to_itrf = transform_position_1(
            position=_TEST_POSITION_I,
            t=_TEST_TIME,
        )

        test_orientation_2 = SpiceOrientation(
            from_frame=ReferenceFrame.get("ITRF"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
        )
        transform_position_2 = test_orientation_2.transform_position

        # Transform position back from ITRF to ICRF_EC
        transformed_back_to_icrf_ec = transform_position_2(
            position=transformed_to_itrf,
            t=_TEST_TIME,
        )

        # Compare the original position to the resulting position
        np.testing.assert_allclose(
            _TEST_POSITION_I.to_numpy(),
            transformed_back_to_icrf_ec.to_numpy(),
            atol=1e-6,
            err_msg="Round-trip transformation failed: ICRF_EC -> ITRF -> ICRF_EC",
        )
        self.assertEqual(
            transformed_back_to_icrf_ec.frame,
            ReferenceFrame.get("ICRF_EC"),
            "The resulting frame is not ICRF_EC after round-trip transformation.",
        )

    def test_transform_position_with_astropy(self):
        """Test transformation using astropy_transform for validation.
        It has been found that the results agree to about a meter accuracy.
        THe difference could be due to the differences in the ICRF (SPICE) and GCRF
        (Astropy) frames."""
        # Validate the position transformation from ICRF_EC to ITRF
        is_valid = validate_transform_position_with_astropy(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
            position=_TEST_POSITION_I,
            time=_TEST_TIME,
        )
        self.assertTrue(is_valid, "Transform position validation failed.")

        # Validate the position transformation from ITRF to ICRF_EC
        is_valid = validate_transform_position_with_astropy(
            from_frame=ReferenceFrame.get("ITRF"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
            position=_TEST_POSITION_EF,
            time=_TEST_TIME,
        )
        self.assertTrue(is_valid, "Transform position validation failed.")


if __name__ == "__main__":
    unittest.main()

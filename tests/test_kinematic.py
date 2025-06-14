"""Unit tests for eosimutils.kinematic module."""

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
from eosimutils.kinematic import transform_position
from eosimutils.kinematic import transform_state

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


class TestTransformPosition(unittest.TestCase):
    """Test the tranform_position function."""

    def test_no_transformation(self):
        """Test no transformation when from_frame and to_frame are the same."""
        transformed_position = transform_position(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
            position=_TEST_POSITION_I,
            time=_TEST_TIME,
        )
        np.testing.assert_array_equal(
            transformed_position.to_list(), _TEST_POSITION_I.to_list()
        )
        self.assertEqual(
            transformed_position.frame, ReferenceFrame.get("ICRF_EC")
        )

    def test_transform_gcrf_to_itrf(self):
        """Test transformation from ICRF_EC to ITRF."""
        transformed_position = transform_position(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
            position=_TEST_POSITION_I,
            time=_TEST_TIME,
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
        with self.assertRaises(NotImplementedError):
            transform_position(
                from_frame="INVALID_FRAME",
                to_frame=ReferenceFrame.get("ITRF"),
                position=_TEST_POSITION_I,
                time=_TEST_TIME,
            )

    def test_round_trip_transformation(self):
        """Test round-trip transformation from ICRF_EC to ITRF and back to ICRF_EC."""
        # Transform position from ICRF_EC to ITRF
        transformed_to_itrf = transform_position(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
            position=_TEST_POSITION_I,
            time=_TEST_TIME,
        )

        # Transform position back from ITRF to ICRF_EC
        transformed_back_to_icrf_ec = transform_position(
            from_frame=ReferenceFrame.get("ITRF"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
            position=transformed_to_itrf,
            time=_TEST_TIME,
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
        This could be due to the differences in the ICRF (SPICE) and GCRF
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


class TestTransformState(unittest.TestCase):
    """Test the transform_state function."""

    def test_no_transformation(self):
        """Test no transformation when from_frame and to_frame are the same."""
        transformed_state = transform_state(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
            state=_TEST_STATE_I,
            time=_TEST_TIME,
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

    def test_transform_gcrf_to_itrf(self):
        """Test transformation from ICRF_EC to ITRF."""
        transformed_state = transform_state(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
            state=_TEST_STATE_I,
            time=_TEST_TIME,
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
        """Test transformation with invalid frames."""
        with self.assertRaises(NotImplementedError):
            transform_state(
                from_frame="INVALID_FRAME",
                to_frame=ReferenceFrame.get("ITRF"),
                state=_TEST_STATE_I,
                time=_TEST_TIME,
            )

    def test_round_trip_transformation(self):
        """Test round-trip transformation from ICRF_EC to ITRF and back to ICRF_EC."""
        # Transform state from ICRF_EC to ITRF
        transformed_to_itrf = transform_state(
            from_frame=ReferenceFrame.get("ICRF_EC"),
            to_frame=ReferenceFrame.get("ITRF"),
            state=_TEST_STATE_I,
            time=_TEST_TIME,
        )

        # Transform state back from ITRF to ICRF_EC
        transformed_back_to_icrf = transform_state(
            from_frame=ReferenceFrame.get("ITRF"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
            state=transformed_to_itrf,
            time=_TEST_TIME,
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
            time=_TEST_TIME,
        )
        self.assertTrue(
            is_valid, "Transform state validation failed for ICRF_EC to ITRF."
        )

        # Validate the state transformation from ITRF to ICRF_EC
        is_valid = validate_transform_state_with_astropy(
            from_frame=ReferenceFrame.get("ITRF"),
            to_frame=ReferenceFrame.get("ICRF_EC"),
            state=_TEST_STATE_EF,
            time=_TEST_TIME,
        )
        self.assertTrue(
            is_valid, "Transform state validation failed for ITRF to ICRF_EC."
        )


if __name__ == "__main__":
    unittest.main()

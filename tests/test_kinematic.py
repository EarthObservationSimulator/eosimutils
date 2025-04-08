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

from testutils import validate_transform_position_with_astropy

class TestTransformPosition(unittest.TestCase):
    """Test the tranform_position function."""

    def setUp(self):
        # Example position in inertial frame
        self.position_i = Cartesian3DPosition(
            random.uniform(-10000, 10000),
            random.uniform(-10000, 10000),
            random.uniform(-10000, 10000),
            ReferenceFrame.EARTH_ICRF,
        )
        # Example position in Earth-fixed frame
        self.position_ef = Cartesian3DPosition(
            random.uniform(-10000, 10000),
            random.uniform(-10000, 10000),
            random.uniform(-10000, 10000),
            ReferenceFrame.EARTH_ICRF,
        )
        # Example time
        year = random.randint(2000, 2030)
        month = random.randint(1, 12)
        day = random.randint(1, 28)  # To avoid invalid dates
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        self.time = AbsoluteDate.from_dict(
            {
            "time_format": "Gregorian_Date",
            "calendar_date": f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}",
            "time_scale": "UTC",
            }
        )

    def test_no_transformation(self):
        """Test no transformation when from_frame and to_frame are the same."""
        transformed_position = transform_position(
            from_frame=ReferenceFrame.EARTH_ICRF,
            to_frame=ReferenceFrame.EARTH_ICRF,
            position=self.position_i,
            time=self.time,
        )
        np.testing.assert_array_equal(
            transformed_position.to_list(), self.position_i.to_list()
        )
        self.assertEqual(transformed_position.frame, ReferenceFrame.EARTH_ICRF)

    def test_transform_gcrf_to_itrf(self):
        """Test transformation from EARTH_ICRF to ITRF."""
        transformed_position = transform_position(
            from_frame=ReferenceFrame.EARTH_ICRF,
            to_frame=ReferenceFrame.ITRF,
            position=self.position_i,
            time=self.time,
        )
        # Validate the transformed position
        self.assertEqual(transformed_position.frame, ReferenceFrame.ITRF)
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
                to_frame=ReferenceFrame.ITRF,
                position=self.position_i,
                time=self.time,
            )
    
    def test_round_trip_transformation(self):
        """Test round-trip transformation from EARTH_ICRF to ITRF and back to EARTH_ICRF."""
        # Transform position from EARTH_ICRF to ITRF
        transformed_to_itrf = transform_position(
            from_frame=ReferenceFrame.EARTH_ICRF,
            to_frame=ReferenceFrame.ITRF,
            position=self.position_i,
            time=self.time,
        )

        # Transform position back from ITRF to EARTH_ICRF
        transformed_back_to_earth_icrf = transform_position(
            from_frame=ReferenceFrame.ITRF,
            to_frame=ReferenceFrame.EARTH_ICRF,
            position=transformed_to_itrf,
            time=self.time,
        )

        # Compare the original position to the resulting position
        np.testing.assert_allclose(
            self.position_i.to_numpy(),
            transformed_back_to_earth_icrf.to_numpy(),
            atol=1e-6,
            err_msg="Round-trip transformation failed: EARTH_ICRF -> ITRF -> EARTH_ICRF",
        )
        self.assertEqual(
            transformed_back_to_earth_icrf.frame,
            ReferenceFrame.EARTH_ICRF,
            "The resulting frame is not EARTH_ICRF after round-trip transformation.",
        )

    def test_transform_position_with_astropy(self):
        """Test transformation using astropy_transform for validation.
        It has been found that the results agree to about a meter accuracy.
        This could be due to the differences in the ICRF (SPICE) and GCRF 
        (Astropy) frames."""
        # Validate the eci-to-ecef transform_position function using astropy_transform
        is_valid = validate_transform_position_with_astropy(
            from_frame=ReferenceFrame.EARTH_ICRF,
            to_frame=ReferenceFrame.ITRF,
            position=self.position_i,
            time=self.time,
        )
        self.assertTrue(is_valid, "Transform position validation failed.")

        # Validate the ecef-to-eci transform_position function using astropy_transform
        is_valid = validate_transform_position_with_astropy(
            from_frame=ReferenceFrame.ITRF,
            to_frame=ReferenceFrame.EARTH_ICRF,
            position=self.position_ef,
            time=self.time,
        )
        self.assertTrue(is_valid, "Transform position validation failed.")
        



if __name__ == "__main__":
    unittest.main()

"""Unit tests for orbitpy.timeseries module."""

# pylint: disable=protected-access

import unittest
import numpy as np
from eosimutils.timeseries import Timeseries
from eosimutils.time import AbsoluteDateArray, JD_OF_J2000


class TestTimeseries(unittest.TestCase):
    """Unit tests for the Timeseries class."""

    def setUp(self):
        self.time = AbsoluteDateArray.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": [JD_OF_J2000 + t for t in [0, 1, 2, 3]],
                "time_scale": "UTC",
            }
        )
        self.data = [np.array([1.0, 2.0, np.nan, 4.0])]
        self.headers = ["example_header"]
        self.timeseries = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in self.data],
                "headers": self.headers,
                "interpolator": "linear",
            }
        )

    def test_resample_data(self):
        """Test the resampling of data with interpolation."""
        new_time = AbsoluteDateArray.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": [JD_OF_J2000 + t for t in [0.5, 1.5, 2.5, 4.5]],
                "time_scale": "UTC",
            }
        )
        resampled_time, resampled_data, _ = self.timeseries._resample_data(
            new_time.ephemeris_time
        )

        np.testing.assert_array_equal(
            resampled_time.ephemeris_time, new_time.ephemeris_time
        )

        # Allclose used since values are interpolated
        np.testing.assert_allclose(
            resampled_data[0], [1.5, np.nan, np.nan, np.nan], equal_nan=True
        )

    def test_remove_gaps_data(self):
        """Test the removal of leading and trailing gaps."""
        data_with_gaps = [np.array([np.nan, 2.0, 3.0, np.nan])]
        ts = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in data_with_gaps],
                "headers": self.headers,
                "interpolator": "linear",
            }
        )
        trimmed_time, trimmed_data, _ = ts._remove_gaps_data()

        np.testing.assert_array_equal(
            trimmed_time.to_dict("JULIAN_DATE")["jd"],
            [JD_OF_J2000 + t for t in [1, 2]],
        )
        np.testing.assert_array_equal(trimmed_data[0], [2.0, 3.0])

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of Timeseries."""
        serialized = self.timeseries.to_dict()
        deserialized = Timeseries.from_dict(serialized)

        # All close used due to rounding in serialization.
        # TODO: May be worth looking into precision of timestrings in the future.
        np.testing.assert_allclose(
            deserialized.time.ephemeris_time,
            self.timeseries.time.ephemeris_time,
            rtol=1e-4,
            atol=1e-4,
        )
        for d1, d2 in zip(deserialized.data, self.timeseries.data):
            np.testing.assert_array_equal(d1, d2)
        self.assertEqual(deserialized.headers, self.timeseries.headers)
        self.assertEqual(
            deserialized.interpolator, self.timeseries.interpolator
        )

    def test_edge_case_empty_data(self):
        """Test edge case where data is empty."""
        empty_time = AbsoluteDateArray.from_dict(
            {"time_format": "Julian_Date", "jd": [], "time_scale": "UTC"}
        )
        empty_data = []
        ts = Timeseries.from_dict(
            {
                "time": empty_time.to_dict("JULIAN_DATE"),
                "data": empty_data,
                "headers": [],
                "interpolator": "linear",
            }
        )

        self.assertEqual(ts.time.ephemeris_time.size, 0)
        self.assertEqual(ts.data, [])

    def test_edge_case_single_point_interpolation(self):
        """Test edge case where only one valid data point exists."""
        single_point_data = [np.array([np.nan, 2.0, np.nan, np.nan])]
        ts = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in single_point_data],
                "headers": self.headers,
                "interpolator": "linear",
            }
        )
        new_time = AbsoluteDateArray.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": [JD_OF_J2000 + t for t in [0.5, 1.5, 2.5]],
                "time_scale": "UTC",
            }
        )
        resampled_time, resampled_data, _ = ts._resample_data(
            new_time.ephemeris_time
        )

        np.testing.assert_array_equal(
            resampled_time.ephemeris_time, new_time.ephemeris_time
        )
        self.assertTrue(np.all(np.isnan(resampled_data[0])))


if __name__ == "__main__":
    unittest.main()

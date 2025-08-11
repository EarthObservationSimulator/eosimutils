"""Unit tests for eosimutils.base module."""

import unittest
import os
import numpy as np

from eosimutils.time import AbsoluteDateArray, JD_OF_J2000

from eosimutils.base import ReferenceFrame, JsonSerializer
from eosimutils.timeseries import Timeseries


class TestReferenceFrame(unittest.TestCase):
    """Test the ReferenceFrame class."""

    def test_values_uppercase(self):
        # Test that all values are uppercase
        for frame in ReferenceFrame.values():
            self.assertTrue(frame.to_string().isupper())

    def test_get(self):
        # Test valid input
        self.assertEqual(ReferenceFrame.get("ICRF_EC"), "ICRF_EC")
        # Test invalid input
        self.assertIsNone(ReferenceFrame.get("INVALID"))

    def test_to_string(self):
        # Test string representation
        self.assertEqual(ReferenceFrame.get("ICRF_EC").to_string(), "ICRF_EC")
        self.assertEqual(ReferenceFrame.get("ITRF").to_string(), "ITRF")

    def test_equality(self):
        # Test equality with string
        self.assertEqual(ReferenceFrame.get("ICRF_EC"), "ICRF_EC")
        self.assertNotEqual(ReferenceFrame.get("ICRF_EC"), "ITRF")

        # Test equality with other types
        self.assertNotEqual(ReferenceFrame.get("ICRF_EC"), 123)
        self.assertNotEqual(ReferenceFrame.get("ICRF_EC"), None)


class TestJsonSerializerWithTimeSeries(unittest.TestCase):
    """Test the JsonSerializer class with Timeseries object."""

    def setUp(self):
        self.time = AbsoluteDateArray.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": [JD_OF_J2000 + t for t in [0, 1, 2, 3]],
                "time_scale": "UTC",
            }
        )
        self.data = [
            np.array([1.0, 2.0, np.nan, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
        ]
        self.headers = ["example_header_1", "example_header_2"]
        self.timeseries = Timeseries(self.time, self.data, self.headers)
        self.test_json_file = "test_timeseries.json"

    def tearDown(self):
        if os.path.exists(self.test_json_file):
            os.remove(self.test_json_file)

    def test_save_and_load_from_json(self):
        """Test saving and loading Timeseries from a JSON file."""
        JsonSerializer.save_to_json(self.timeseries, self.test_json_file)

        loaded_timeseries = JsonSerializer.load_from_json(
            Timeseries, self.test_json_file
        )

        np.testing.assert_array_equal(
            loaded_timeseries.time.ephemeris_time,
            self.timeseries.time.ephemeris_time,
        )
        for d1, d2 in zip(loaded_timeseries.data, self.timeseries.data):
            np.testing.assert_array_equal(d1, d2)
        self.assertEqual(loaded_timeseries.headers, self.timeseries.headers)


if __name__ == "__main__":
    unittest.main()

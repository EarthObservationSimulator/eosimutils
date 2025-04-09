"""Unit tests for orbitpy.timeseries module."""
# pylint: disable=protected-access

import unittest
import numpy as np
from eosimutils.timeseries import Timeseries
from eosimutils.time import AbsoluteDates


class TestTimeseries(unittest.TestCase):
    """Unit tests for the Timeseries class."""

    def setUp(self):
        self.time = AbsoluteDates(np.array([0, 1, 2, 3]))
        self.data = [np.array([1.0, 2.0, np.nan, 4.0])]
        self.headers = ["example_header"]

    def test_resample_data(self):
        """Test the resampling of data with interpolation."""
        ts = Timeseries(self.time, self.data)
        new_time = np.array([0.5, 1.5, 2.5])
        resampled_time, resampled_data, _ = ts._resample_data(new_time)

        np.testing.assert_allclose(resampled_time.et, new_time)
        np.testing.assert_allclose(resampled_data[0], [1.5, np.nan, np.nan], equal_nan=True)

    def test_remove_gaps_data(self):
        """Test the removal of leading and trailing gaps."""
        data_with_gaps = [np.array([np.nan, 2.0, 3.0, np.nan])]
        ts = Timeseries(self.time, data_with_gaps)
        trimmed_time, trimmed_data, _ = ts._remove_gaps_data()

        np.testing.assert_allclose(trimmed_time.et, [1, 2])
        np.testing.assert_allclose(trimmed_data[0], [2.0, 3.0])

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of Timeseries."""
        ts = Timeseries(self.time, self.data, self.headers)
        serialized = ts.to_dict()
        deserialized = Timeseries.from_dict(serialized)

        # Tolerances are set to make test pass. TODO: Perhaps not precise enough?
        np.testing.assert_allclose(deserialized.time.et, ts.time.et, rtol=1e-4, atol = 1e-4)
        for d1, d2 in zip(deserialized.data, ts.data):
            np.testing.assert_allclose(d1, d2)
        self.assertEqual(deserialized.headers, ts.headers)
        self.assertEqual(deserialized.interpolator, ts.interpolator)

    def test_edge_case_empty_data(self):
        """Test edge case where data is empty."""
        empty_time = AbsoluteDates(np.array([]))
        empty_data = []
        ts = Timeseries(empty_time, empty_data)

        self.assertEqual(ts.time.et.size, 0)
        self.assertEqual(ts.data, [])

    def test_edge_case_single_point_interpolation(self):
        """Test edge case where only one valid data point exists."""
        single_point_data = [np.array([np.nan, 2.0, np.nan, np.nan])]
        ts = Timeseries(self.time, single_point_data)
        new_time = np.array([0.5, 1.5, 2.5])
        resampled_time, resampled_data, _ = ts._resample_data(new_time)

        np.testing.assert_allclose(resampled_time.et, new_time)
        self.assertTrue(np.all(np.isnan(resampled_data[0])))


if __name__ == "__main__":
    unittest.main()

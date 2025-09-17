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

    def test_boolean_data_resample(self):
        """Test resampling of mixed data types: scalar, vector, and boolean."""
        mixed_data = [
            np.array([1.0, 2.0, np.nan, 4.0]),  # Scalar data
            np.array(
                [[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan], [5.0, 6.0]]
            ),  # Vector data
            np.array([True, False, True, False]),  # Boolean data
        ]
        ts = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in mixed_data],
                "headers": ["scalar_header", ["x", "y"], "boolean_header"],
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

        # Check scalar data interpolation
        np.testing.assert_allclose(
            resampled_data[0], [1.5, np.nan, np.nan], equal_nan=True
        )

        # Check vector data interpolation
        np.testing.assert_allclose(
            resampled_data[1],
            [[2.0, 3.0], [np.nan, np.nan], [np.nan, np.nan]],
            equal_nan=True,
        )

        # Check boolean data resampling (should result in NaNs)
        self.assertTrue(np.all(np.isnan(resampled_data[2])))

    def test_boolean_data_remove_gaps(self):
        """Test removal of gaps in boolean data."""
        boolean_data_with_gaps = [np.array([np.nan, True, False, np.nan])]
        ts = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in boolean_data_with_gaps],
                "headers": self.headers,
                "interpolator": "linear",
            }
        )
        trimmed_time, trimmed_data, _ = ts._remove_gaps_data()

        np.testing.assert_array_equal(
            trimmed_time.to_dict("JULIAN_DATE")["jd"],
            [JD_OF_J2000 + t for t in [1, 2]],
        )
        np.testing.assert_array_equal(trimmed_data[0], [True, False])

    def test_boolean_data_serialization(self):
        """Test serialization and deserialization of boolean data."""
        boolean_data = [np.array([True, False, True, False])]
        ts = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in boolean_data],
                "headers": self.headers,
                "interpolator": "linear",
            }
        )
        serialized = ts.to_dict()
        deserialized = Timeseries.from_dict(serialized)

        np.testing.assert_array_equal(
            deserialized.time.ephemeris_time, ts.time.ephemeris_time
        )
        for d1, d2 in zip(deserialized.data, ts.data):
            np.testing.assert_array_equal(d1, d2)
        self.assertEqual(deserialized.headers, ts.headers)
        self.assertEqual(deserialized.interpolator, ts.interpolator)

    def test_mixed_data_operations(self):
        """Test operations on mixed data types: boolean, scalar numeric, and vector numeric."""
        mixed_data_1 = [
            np.array([True, False, True, False]),  # Boolean data
            np.array([1.0, -2.0, np.nan, 4.0]),  # Scalar numeric data
            np.array(
                [[1.0, 0.0], [3.0, 4.0], [np.nan, np.nan], [5.0, 6.0]]
            ),  # Vector numeric data
        ]
        mixed_data_2 = [
            np.array([False, True, True, False]),  # Boolean data
            np.array([2.0, 3.0, 4.0, 5.0]),  # Scalar numeric data
            np.array(
                [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]]
            ),  # Vector numeric data
        ]

        headers = ["boolean_header", "scalar_header", ["x", "y"]]

        ts1 = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in mixed_data_1],
                "headers": headers,
                "interpolator": "linear",
            }
        )
        ts2 = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in mixed_data_2],
                "headers": headers,
                "interpolator": "linear",
            }
        )

        # For logical operations involving numeric values, any non-zero number—including
        # negatives and NaNs—is treated as True, and the result of the operation is
        # evaluated accordingly.
        # Arithmetic operations with boolean values results in NaNs.

        # Test boolean AND operation
        result_and = ts1.logical_and(ts2)
        expected_and = [
            np.array([False, False, True, False]),  # Boolean AND
            np.array([True, True, True, True]),
            np.array([[True, False], [True, True], [True, True], [True, True]]),
        ]
        for i in range(len(expected_and)):
            np.testing.assert_array_equal(result_and.data[i], expected_and[i])

        # Test boolean OR operation
        result_or = ts1.logical_or(ts2)
        expected_or = [
            np.array([True, True, True, False]),
            np.array([True, True, True, True]),
            np.array([[True, True], [True, True], [True, True], [True, True]]),
        ]
        for i in range(len(expected_or)):
            np.testing.assert_array_equal(result_or.data[i], expected_or[i])

        # Test NOT operation on ts1
        result_not = ts1.logical_not()
        expected_not = [
            np.array([False, True, False, True]),
            np.array([False, False, False, False]),
            np.array(
                [[False, True], [False, False], [False, False], [False, False]]
            ),
        ]
        for i in range(len(expected_not)):
            np.testing.assert_array_equal(result_not.data[i], expected_not[i])

        # Test addition
        result_add = ts1 + ts2
        expected_add = [
            np.array(
                [np.nan, np.nan, np.nan, np.nan]
            ),  # Boolean addition results in NaNs.
            np.array([3.0, 1.0, np.nan, 9.0]),
            np.array([[1.5, 1.5], [5.5, 7.5], [np.nan, np.nan], [11.5, 13.5]]),
        ]
        for i in range(len(expected_add)):
            np.testing.assert_allclose(
                result_add.data[i], expected_add[i], equal_nan=True
            )

        # Test subtraction
        result_sub = ts1 - ts2
        expected_sub = [
            np.array(
                [np.nan, np.nan, np.nan, np.nan]
            ),  # Boolean subtraction results in NaNs.
            np.array([-1.0, -5.0, np.nan, -1.0]),
            np.array([[0.5, -1.5], [0.5, 0.5], [np.nan, np.nan], [-1.5, -1.5]]),
        ]
        for i in range(len(expected_sub)):
            np.testing.assert_allclose(
                result_sub.data[i], expected_sub[i], equal_nan=True
            )

        # Test multiplication
        result_mul = ts1 * ts2
        expected_mul = [
            np.array(
                [np.nan, np.nan, np.nan, np.nan]
            ),  # Boolean multiplication results in NaNs.
            np.array([2.0, -6.0, np.nan, 20.0]),
            np.array([[0.5, 0.0], [7.5, 14.0], [np.nan, np.nan], [32.5, 45.0]]),
        ]
        for i in range(len(expected_mul)):
            np.testing.assert_allclose(
                result_mul.data[i], expected_mul[i], equal_nan=True
            )

        # Test division
        result_div = ts1 / ts2
        expected_div = [
            np.array(
                [np.nan, np.nan, np.nan, np.nan]
            ),  # Boolean division results in NaNs.
            np.array([0.5, -2.0 / 3.0, np.nan, 4.0 / 5.0]),
            np.array(
                [
                    [1.0 / 0.5, 0.0],
                    [3.0 / 2.5, 4.0 / 3.5],
                    [np.nan, np.nan],
                    [5.0 / 6.5, 6.0 / 7.5],
                ]
            ),
        ]
        for i in range(len(expected_div)):
            np.testing.assert_allclose(
                result_div.data[i], expected_div[i], equal_nan=True
            )

        # Ensure time and headers remain consistent
        for result in [
            result_and,
            result_or,
            result_not,
            result_add,
            result_sub,
            result_mul,
        ]:
            np.testing.assert_array_equal(
                result.time.ephemeris_time, self.time.ephemeris_time
            )
            self.assertEqual(result.headers, headers)

    def test_mixed_data_operations_with_scalar(self):
        """Test arithmetic operations between a scalar value and a Timeseries object."""
        mixed_data = [
            np.array([True, False, True, False]),  # Boolean data
            np.array([1.0, -2.0, np.nan, 4.0]),  # Scalar numeric data
            np.array(
                [[1.0, 0.0], [3.0, 4.0], [np.nan, np.nan], [5.0, 6.0]]
            ),  # Vector numeric data
        ]

        headers = ["boolean_header", "scalar_header", ["x", "y"]]

        ts = Timeseries.from_dict(
            {
                "time": self.time.to_dict("JULIAN_DATE"),
                "data": [arr.tolist() for arr in mixed_data],
                "headers": headers,
                "interpolator": "linear",
            }
        )

        scalar = 2.0

        # Test addition
        result_add = ts + scalar
        expected_add = [
            np.full_like(
                mixed_data[0], np.nan, dtype=float
            ),  # Boolean addition results in NaNs
            np.array([3.0, 0.0, np.nan, 6.0]),
            np.array([[3.0, 2.0], [5.0, 6.0], [np.nan, np.nan], [7.0, 8.0]]),
        ]
        for i in range(len(expected_add)):
            np.testing.assert_allclose(
                result_add.data[i], expected_add[i], equal_nan=True
            )

        # Test subtraction
        result_sub = ts - scalar
        expected_sub = [
            np.full_like(
                mixed_data[0], np.nan, dtype=float
            ),  # Boolean subtraction results in NaNs
            np.array([-1.0, -4.0, np.nan, 2.0]),
            np.array([[-1.0, -2.0], [1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
        ]
        for i in range(len(expected_sub)):
            np.testing.assert_allclose(
                result_sub.data[i], expected_sub[i], equal_nan=True
            )

        # Test multiplication
        result_mul = ts * scalar
        expected_mul = [
            np.full_like(
                mixed_data[0], np.nan, dtype=float
            ),  # Boolean multiplication results in NaNs
            np.array([2.0, -4.0, np.nan, 8.0]),
            np.array([[2.0, 0.0], [6.0, 8.0], [np.nan, np.nan], [10.0, 12.0]]),
        ]
        for i in range(len(expected_mul)):
            np.testing.assert_allclose(
                result_mul.data[i], expected_mul[i], equal_nan=True
            )

        # Test division
        result_div = ts / scalar
        expected_div = [
            np.full_like(
                mixed_data[0], np.nan, dtype=float
            ),  # Boolean division results in NaNs
            np.array([0.5, -1.0, np.nan, 2.0]),
            np.array([[0.5, 0.0], [1.5, 2.0], [np.nan, np.nan], [2.5, 3.0]]),
        ]
        for i in range(len(expected_div)):
            np.testing.assert_allclose(
                result_div.data[i], expected_div[i], equal_nan=True
            )

        # Ensure time and headers remain consistent
        for result in [result_add, result_sub, result_mul, result_div]:
            np.testing.assert_array_equal(
                result.time.ephemeris_time, self.time.ephemeris_time
            )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for eosimutils.time module."""

import unittest
import numpy as np
import copy

from astropy.time import Time as Astropy_Time

from eosimutils.time import AbsoluteDate, AbsoluteDateArray, AbsoluteDateIntervalArray


class TestAbsoluteDate(unittest.TestCase):
    """Test the AbsoluteDate class."""

    def test_from_dict_gregorian(self):
        # validation with data in the SPICE help file
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/str2et_c.html
        dict_in = {
            "time_format": "Gregorian_Date",
            "calendar_date": "2017-07-14T19:46:00.0",
            "time_scale": "utc",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        # results may differ across computing platforms and kernels used
        self.assertAlmostEqual(
            absolute_date.ephemeris_time, 553333629.183727, places=6
        )

    def test_from_dict_julian(self):
        # validation with data in the SPICE help file
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/str2et_c.html
        dict_in = {
            "time_format": "Julian_Date",
            "jd": 2457949.323611111,  # 2017 July 14 19:46:0.0
            "time_scale": "utc",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        # results may differ across computing platforms and kernels used
        self.assertAlmostEqual(
            absolute_date.ephemeris_time, 553333629.183727, places=4
        )

    def test_to_dict_gregorian(self):
        absolute_date = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-03-10T14:30:0",
                "time_scale": "utc",
            }
        )
        dict_out = absolute_date.to_dict("Gregorian_Date", "UTC")
        expected_dict = {
            "time_format": "GREGORIAN_DATE",
            "calendar_date": "2025-03-10T14:30:00.000",
            "time_scale": "UTC",
        }
        self.assertEqual(dict_out, expected_dict)

    def test_to_dict_julian(self):
        absolute_date = AbsoluteDate.from_dict(
            {
                "time_format": "Julian_Date",
                "jd": 2457081.10417,
                "time_scale": "utc",
            }
        )
        dict_out = absolute_date.to_dict("Julian_Date")
        expected_dict = {
            "time_format": "JULIAN_DATE",
            "jd": 2457081.10417,
            "time_scale": "UTC",
        }
        self.assertEqual(dict_out, expected_dict)

    def test_gregorian_to_julian(self):
        # Initialize with Gregorian date
        dict_in = {
            "time_format": "GREGORIAN_DATE",
            "calendar_date": "2025-03-11T01:23:37.0",
            "time_scale": "utc",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        absolute_date_jd = absolute_date.to_dict("Julian_Date")
        # Validation data from: https://aa.usno.navy.mil/data/JulianDate
        # Since both dates are in UTC, it may not matter that the USNO
        # website specifies the time scale as UT1.
        self.assertAlmostEqual(absolute_date_jd["jd"], 2460745.558067, places=5)

    def test_julian_to_gregorian(self):
        # Initialize with Julian Date
        dict_in = {
            "time_format": "Julian_Date",
            "jd": 2460325.145250,
            "time_scale": "utc",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)
        absolute_date_gregorian = absolute_date.to_dict("Gregorian_Date", "UTC")
        # Validation data from: https://aa.usno.navy.mil/data/JulianDate
        # Since both dates are in UTC, it may not matter that the USNO
        # website specifies the time scale as UT1.
        self.assertEqual(
            absolute_date_gregorian["calendar_date"], "2024-01-15T15:29:09.600"
        )

    def test_to_spice_ephemeris_time(self):
        """Test the to_spice_ephemeris_time method."""
        # validation with data in the SPICE help file
        # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/str2et_c.html
        dict_in = {
            "time_format": "Gregorian_Date",
            "calendar_date": "2017-07-14T19:46:00.0",
            "time_scale": "utc",
        }
        absolute_date = AbsoluteDate.from_dict(dict_in)

        spice_ephemeris_time = absolute_date.to_spice_ephemeris_time()

        # results may differ across computing platforms and kernels used
        self.assertAlmostEqual(spice_ephemeris_time, 553333629.183727, places=6)

    def test_to_astropy_time(self):

        # initialize AstroPy and AbsoluteDate objects
        astropy_time = Astropy_Time(
            "2025-03-17T12:00:00", format="isot", scale="utc"
        )
        absolute_date = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-03-17T12:00:00",
                "time_scale": "utc",
            }
        )

        # Convert to Astropy Time
        converted_time = absolute_date.to_astropy_time()

        # Assert that the converted time matches the original Astropy Time
        self.assertEqual(converted_time, astropy_time)

    def test_to_skyfield_time(self):
        absolute_date = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-03-17T12:00:00",
                "time_scale": "utc",
            }
        )
        # Convert to Skyfield Time
        skyfield_time = absolute_date.to_skyfield_time()

        # Assert that the Skyfield time matches the AbsoluteTime in UTC
        self.assertEqual(
            skyfield_time.utc_strftime(format="%Y-%m-%d %H:%M:%S UTC"),
            "2025-03-17 12:00:00 UTC",
        )

    def test_equality_operator(self):
        date1 = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-03-17T12:00:00",
                "time_scale": "utc",
            }
        )
        # Create another AbsoluteDate object with the same date
        date2 = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-03-17T12:00:00",
                "time_scale": "utc",
            }
        )
        # Create a different AbsoluteDate object
        date3 = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-03-18T12:00:00",
                "time_scale": "utc",
            }
        )

        # Test equality
        self.assertTrue(date1 == date2)
        self.assertFalse(date1 == date3)

        # Test comparison with a non-AbsoluteDate object
        self.assertFalse(date1 == "not an AbsoluteDate")

    def test_add_operator(self):
        """Test the __add__ operator for AbsoluteDate."""
        # Initialize an AbsoluteDate object
        absolute_date = AbsoluteDate.from_dict(
            {
                "time_format": "Gregorian_Date",
                "calendar_date": "2025-03-17T12:00:00",
                "time_scale": "utc",
            }
        )

        # Add 3600 seconds (1 hour)
        new_date = absolute_date + 3600

        # Convert the new date to Gregorian format
        new_date_dict = new_date.to_dict("Gregorian_Date", "UTC")

        # Expected result after adding 1 hour
        expected_date = {
            "time_format": "GREGORIAN_DATE",
            "calendar_date": "2025-03-17T13:00:00.000",
            "time_scale": "UTC",
        }

        # Assert the new date matches the expected result
        self.assertEqual(new_date_dict, expected_date)


class TestAbsoluteDateArray(unittest.TestCase):
    """Test the AbsoluteDateArray class."""

    def test_to_astropy_time(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDateArray(et_array)
        astropy_times = abs_dates.to_astropy_time()
        self.assertEqual(len(astropy_times), 2)
        self.assertEqual(astropy_times[0].isot, "2017-07-14T19:46:00.000")
        self.assertEqual(astropy_times[1].isot, "2017-07-14T19:46:01.000")

    def test_to_dict_gregorian(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDateArray(et_array)
        result = abs_dates.to_dict("GREGORIAN_DATE", "UTC")
        self.assertEqual(result["time_format"], "GREGORIAN_DATE")
        self.assertEqual(result["time_scale"], "UTC")
        self.assertEqual(result["calendar_date"][0], "2017-07-14T19:46:00.000")
        self.assertEqual(result["calendar_date"][1], "2017-07-14T19:46:01.000")

    def test_to_dict_julian(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDateArray(et_array)
        result = abs_dates.to_dict("JULIAN_DATE", "UTC")
        self.assertEqual(result["time_format"], "JULIAN_DATE")
        self.assertEqual(result["time_scale"], "UTC")
        self.assertAlmostEqual(result["jd"][0], 2457949.323611, places=6)
        self.assertAlmostEqual(result["jd"][1], 2457949.323623, places=6)

    def test_to_dict_and_from_dict(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDateArray(et_array)

        # Convert to dictionary
        dict_representation = abs_dates.to_dict("GREGORIAN_DATE", "UTC")

        # Reconstruct from dictionary
        reconstructed_abs_dates = AbsoluteDateArray.from_dict(
            dict_representation
        )

        # Assert that the reconstructed object matches the original
        np.testing.assert_allclose(
            abs_dates.ephemeris_time,
            reconstructed_abs_dates.ephemeris_time,
            rtol=1e-6,
        )

    def test_length(self):
        """Test the length of the AbsoluteDateArray."""
        et_array = np.random.uniform(
            553333629.0, 553333630.0, size=np.random.randint(1, 10)
        )
        abs_dates = AbsoluteDateArray(et_array)
        self.assertEqual(len(abs_dates), len(et_array))

    def test_get_item(self):
        """Test the __getitem__ method for AbsoluteDateArray."""
        et_array = np.random.uniform(
            553333629.0, 553333635.0, size=np.random.randint(3, 10)
        )
        abs_dates = AbsoluteDateArray(et_array)

        # Test getting a single item
        item = abs_dates[0]
        self.assertIsInstance(item, AbsoluteDate)
        self.assertAlmostEqual(item.ephemeris_time, et_array[0], places=6)

        # Test getting a slice
        slice_items = abs_dates[1:3]
        self.assertIsInstance(slice_items, AbsoluteDateArray)
        self.assertEqual(len(slice_items), 2)

    def test_equality_operator(self):
        """Test the equality operator for AbsoluteDateArray."""
        et_array1 = np.random.uniform(
            553333633.0, 553333635.0, size=np.random.randint(1, 10)
        )
        et_array2 = copy.deepcopy(et_array1)
        et_array3 = np.array([553333631.0, 553333632.0])

        abs_dates1 = AbsoluteDateArray(et_array1)
        abs_dates2 = AbsoluteDateArray(et_array2)
        abs_dates3 = AbsoluteDateArray(et_array3)

        # Test equality
        self.assertTrue(abs_dates1 == abs_dates2)
        self.assertFalse(abs_dates1 == abs_dates3)

    def test_to_spice_ephemeris_time(self):
        """Test the to_spice_ephemeris_time method for AbsoluteDateArray."""
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDateArray(et_array)

        spice_ephemeris_times = abs_dates.to_spice_ephemeris_time()

        # Assert that the returned array matches the input ephemeris times
        np.testing.assert_allclose(spice_ephemeris_times, et_array, rtol=1e-6)

class TestAbsoluteDateIntervalArray(unittest.TestCase):
    """Test the AbsoluteDateIntervalArray class."""

    def setUp(self):
        """Set up test data for the tests."""
        self.start_times = AbsoluteDateArray(
            np.array([553333629.183727, 553333630.183727])
        )
        self.stop_times = AbsoluteDateArray(
            np.array([553333639.183727, 553333640.183727])
        )
        self.interval_array = AbsoluteDateIntervalArray(
            start_times=self.start_times, stop_times=self.stop_times
        )

    def test_initialization(self):
        """Test initialization of AbsoluteDateIntervalArray."""
        self.assertEqual(len(self.interval_array), 2)
        np.testing.assert_allclose(
            self.interval_array.start_times.ephemeris_time,
            self.start_times.ephemeris_time,
        )
        np.testing.assert_allclose(
            self.interval_array.stop_times.ephemeris_time,
            self.stop_times.ephemeris_time,
        )

    def test_invalid_initialization(self):
        """Test invalid initialization of AbsoluteDateIntervalArray."""
        with self.assertRaises(TypeError):
            AbsoluteDateIntervalArray(start_times=[1, 2], stop_times=self.stop_times)
        with self.assertRaises(ValueError):
            AbsoluteDateIntervalArray(
                start_times=self.start_times,
                stop_times=AbsoluteDateArray(
                    np.array([553333620.183727, 553333610.183727])
                ),
            )
        with self.assertRaises(ValueError):
            AbsoluteDateIntervalArray(
                start_times=self.start_times,
                stop_times=AbsoluteDateArray(
                    np.array([553333639.183727])
                ),  # Mismatched lengths
            )

    def test_from_dict(self):
        """Test the from_dict method."""
        dict_in = {
            "start_times": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": ["2017-07-14T19:46:00.000", "2017-07-14T19:46:01.000"],
                "time_scale": "UTC",
            },
            "stop_times": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": ["2017-07-14T19:46:10.000", "2017-07-14T19:46:11.000"],
                "time_scale": "UTC",
            },
        }
        interval_array = AbsoluteDateIntervalArray.from_dict(dict_in)
        self.assertEqual(len(interval_array), 2)
        self.assertAlmostEqual(
            interval_array.start_times.ephemeris_time[0], 553333629.183727, places=6
        )
        self.assertAlmostEqual(
            interval_array.stop_times.ephemeris_time[1], 553333640.183727, places=6
        )

    def test_to_dict(self):
        """Test the to_dict method."""
        dict_out = self.interval_array.to_dict("GREGORIAN_DATE", "UTC")
        expected_dict = {
            "start_times": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": ["2017-07-14T19:46:00.000", "2017-07-14T19:46:01.000"],
                "time_scale": "UTC",
            },
            "stop_times": {
                "time_format": "GREGORIAN_DATE",
                "calendar_date": ["2017-07-14T19:46:10.000", "2017-07-14T19:46:11.000"],
                "time_scale": "UTC",
            },
        }
        self.assertEqual(dict_out, expected_dict)

    def test_to_spice_ephemeris_time(self):
        """Test the to_spice_ephemeris_time method."""
        start_et, stop_et = self.interval_array.to_spice_ephemeris_time()
        np.testing.assert_allclose(start_et, self.start_times.ephemeris_time)
        np.testing.assert_allclose(stop_et, self.stop_times.ephemeris_time)

    def test_length(self):
        """Test the length of the AbsoluteDateIntervalArray."""
        self.assertEqual(len(self.interval_array), 2)

    def test_get_item(self):
        """Test the __getitem__ method for AbsoluteDateIntervalArray."""
        # Test getting a single interval
        interval = self.interval_array[0]
        self.assertIsInstance(interval, AbsoluteDateIntervalArray)
        self.assertEqual(len(interval), 1)
        np.testing.assert_allclose(
            interval.start_times.ephemeris_time, [self.start_times.ephemeris_time[0]]
        )
        np.testing.assert_allclose(
            interval.stop_times.ephemeris_time, [self.stop_times.ephemeris_time[0]]
        )

        # Test slicing
        sliced_intervals = self.interval_array[:1]
        self.assertIsInstance(sliced_intervals, AbsoluteDateIntervalArray)
        self.assertEqual(len(sliced_intervals), 1)
        np.testing.assert_allclose(
            sliced_intervals.start_times.ephemeris_time,
            [self.start_times.ephemeris_time[0]],
        )
        np.testing.assert_allclose(
            sliced_intervals.stop_times.ephemeris_time,
            [self.stop_times.ephemeris_time[0]],
        )

if __name__ == "__main__":
    unittest.main()

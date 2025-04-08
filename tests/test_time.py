"""Unit tests for orbitpy.time module."""

import unittest
import numpy as np

from astropy.time import Time as Astropy_Time

from eosimutils.time import AbsoluteDate, AbsoluteDates


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
        # results may differ across platforms and kernels used
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


class TestAbsoluteDates(unittest.TestCase):
    """Test the AbsoluteDates class."""

    def test_to_astropy_time(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDates(et_array)
        astropy_times = abs_dates.to_astropy_time()
        self.assertEqual(len(astropy_times), 2)
        self.assertEqual(astropy_times[0].isot, "2017-07-14T19:46:00.000")
        self.assertEqual(astropy_times[1].isot, "2017-07-14T19:46:01.000")

    def test_to_dict_gregorian(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDates(et_array)
        result = abs_dates.to_dict("GREGORIAN_DATE", "UTC")
        self.assertEqual(result["time_format"], "GREGORIAN_DATE")
        self.assertEqual(result["time_scale"], "UTC")
        self.assertEqual(result["times"][0], "2017-07-14T19:46:00.000")
        self.assertEqual(result["times"][1], "2017-07-14T19:46:01.000")

    def test_to_dict_julian(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDates(et_array)
        result = abs_dates.to_dict("JULIAN_DATE", "UTC")
        self.assertEqual(result["time_format"], "JULIAN_DATE")
        self.assertEqual(result["time_scale"], "UTC")
        self.assertAlmostEqual(result["times"][0], 2457949.323611, places=6)
        self.assertAlmostEqual(result["times"][1], 2457949.323623, places=6)

    def test_to_dict_and_from_dict(self):
        et_array = np.array([553333629.183727, 553333630.183727])
        abs_dates = AbsoluteDates(et_array)

        # Convert to dictionary
        dict_representation = abs_dates.to_dict("GREGORIAN_DATE", "UTC")

        # Reconstruct from dictionary
        reconstructed_abs_dates = AbsoluteDates.from_dict(dict_representation)

        # Assert that the reconstructed object matches the original
        np.testing.assert_allclose(abs_dates.et, reconstructed_abs_dates.et, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

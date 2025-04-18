"""
.. module:: eosimutils.time
   :synopsis: Time information.

Collection of classes and functions for handling time information.

Constants:
    JD_OF_J2000 (float): Julian Date of the J2000 epoch (2451545.0).
"""

from typing import Dict, Any, Union

import numpy as np
import spiceypy as spice

from eosimutils.spicekernels import load_spice_kernels
from astropy.time import Time as Astropy_Time
from skyfield.api import load as Skyfield_Load
from skyfield.timelib import Time as Skyfield_Time

from .base import EnumBase

# Julian Date of the J2000 epoch
JD_OF_J2000 = 2451545.0


class TimeFormat(EnumBase):
    """
    Enumeration of recognized time formats.
    """

    GREGORIAN_DATE = "GREGORIAN_DATE"
    JULIAN_DATE = "JULIAN_DATE"


class TimeScale(EnumBase):
    """
    Enumeration of recognized time scales.
    """

    UTC = "UTC"


class AbsoluteDate:
    """Handles date-time information with support to Julian and Gregorian
    date-time formats and UTC time scale. Date-time is stored
    internally as Ephemeris Time (ET) (Barycentric Dynamical Time (TDB))
    as defined in SPICE."""

    def __init__(self, ephemeris_time: float) -> None:
        """Constructor for the AbsoluteDate class.

        Args:
            ephemeris_time (float): Ephemeris Time (ET) /
                            Barycentric Dynamical Time (TDB)
        """
        self.ephemeris_time = ephemeris_time

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "AbsoluteDate":
        """Construct an AbsoluteDate object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the date-time information.
                The dictionary should contain the following key-value pairs:
                - "time_format" (str): The date-time format, either
                                       "Gregorian_Date" or "Julian_Date"
                                       (case-insensitive).
                - "time_scale" (str): The time scale, e.g., "UTC"
                                      (case-insensitive).
                                      See :class:`eosimutils.time.TimeScale` for options.

                For "Gregorian_Date" format:
                - "calendar_date" (str): The date-time in YYYY-MM-DDTHH:MM:SS.SSS format.
                                         (e.g., "2025-03-31T12:34:56.789").

                For "Julian_Date" format:
                - "jd" (float): The Julian Date.

        Returns:
            AbsoluteDate: AbsoluteDate object.
        """
        time_scale: TimeScale = TimeScale.get(dict_in["time_scale"])
        time_format: TimeFormat = TimeFormat.get(dict_in["time_format"])

        # Load SPICE kernel files
        load_spice_kernels()

        if time_scale == TimeScale.UTC:

            if time_format == TimeFormat.GREGORIAN_DATE:
                # Parse the calendar date string and convert to Ephemeris Time (ET)
                calendar_date_str: str = dict_in["calendar_date"]
                spice_ephemeris_time = spice.str2et(calendar_date_str)

            elif time_format == TimeFormat.JULIAN_DATE:
                # Convert Julian Date UTC to ET
                jd: float = dict_in["jd"]
                time_string = f"jd {jd}"  # Format as Julian Date string
                spice_ephemeris_time = spice.str2et(time_string)

            else:
                raise ValueError(f"Unsupported date-time format: {time_format}")
        else:
            raise ValueError(f"Unsupported time scale: {time_scale}.")

        return cls(ephemeris_time=spice_ephemeris_time)

    def to_dict(
        self,
        time_format: Union[TimeFormat, str] = "GREGORIAN_DATE",
        time_scale: Union[TimeScale, str] = "UTC",
    ) -> Dict[str, Any]:
        """Convert the AbsoluteDate object to a dictionary.

        Args:
            time_format (str): The type of date-time format to use
                                ("GREGORIAN_DATE" or "JULIAN_DATE").
                                Default is "GREGORIAN_DATE".

            time_scale (str): The time scale to use (e.g., "UTC").
                                Default is "UTC".


        Returns:
            dict: Dictionary with the date-time information.
        """

        # Load SPICE kernel files
        load_spice_kernels()

        time_format = TimeFormat.get(time_format)
        time_scale = TimeScale.get(time_scale)

        if time_scale == TimeScale.UTC:

            if time_format == TimeFormat.GREGORIAN_DATE:

                # Convert Ephemeris Time (ET) to Gregorian Date UTC
                time_string = spice.et2utc(self.ephemeris_time, "ISOC", 3)
                return {
                    "time_format": "GREGORIAN_DATE",
                    "calendar_date": time_string,
                    "time_scale": time_scale.to_string(),
                }
            elif time_format == TimeFormat.JULIAN_DATE:

                # Convert Ephemeris Time (ET) to Julian Date UTC
                time_string = spice.et2utc(self.ephemeris_time, "J", 7)
                # Parse the Julian Date value
                jd_value = float(
                    time_string.split()[1]
                )  # Extract and convert the value to float

                return {
                    "time_format": "JULIAN_DATE",
                    "jd": jd_value,
                    "time_scale": time_scale.to_string(),
                }
            else:
                raise ValueError(f"Unsupported date-time format: {time_format}")
        else:
            raise ValueError(f"Unsupported time scale: {time_scale}.")

    def to_astropy_time(self) -> Astropy_Time:
        """Convert the AbsoluteDate object to an Astropy Time object.

        Returns:
            astropy.time.Time: Astropy Time object.
        """
        gregorian_utc_time_string = spice.et2utc(self.ephemeris_time, "ISOC", 3)
        return Astropy_Time(gregorian_utc_time_string, scale="utc")

    def to_skyfield_time(self) -> Skyfield_Time:
        """Convert the AbsoluteDate object to a Skyfield Time object.

        Returns:
            skyfield.time.Time: Skyfield Time object.
        """
        ts = Skyfield_Load.timescale()
        gregorian_utc_time_string = spice.et2utc(self.ephemeris_time, "ISOC", 3)
        date_part, time_part = gregorian_utc_time_string.split("T")
        year, month, day = map(int, date_part.split("-"))
        hour, minute = map(int, time_part.split(":")[:2])
        second = int(
            float(time_part.split(":")[2])
        )  # Handle fractional seconds
        skyfield_time = ts.utc(
            year, month=month, day=day, hour=hour, minute=minute, second=second
        )
        return skyfield_time

    def to_spice_ephemeris_time(self) -> float:
        """Convert the AbsoluteDate object to a SPICE Ephemeris Time (ET).
        In the SPICE toolkit, ET Means TDB.

        Returns:
            float: Ephemeris Time (ET).
        """
        return self.ephemeris_time

    def __eq__(self, value):
        """Check equality of two AbsoluteDate objects.

        Args:
            value (AbsoluteDate): The AbsoluteDate object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(value, AbsoluteDate):
            return False
        return self.ephemeris_time == value.ephemeris_time


class AbsoluteDateArray:
    """
    Vectorized representation of time in Ephemeris Time (ET).

    This class stores a set of time points as a 1D numpy array (in Ephemeris Time, ET)
    for efficiency. It provides vectorized methods to convert to other time representations,
    such as Astropy Time objects and Skyfield Time objects, as well as exporting time information
    to a dictionary.

    Attributes:
        et (np.ndarray): 1D numpy array of ephemeris times.
    """

    def __init__(self, et: np.ndarray) -> None:
        """
        Constructor for the AbsoluteDateArray class.

        Args:
            et (np.ndarray): 1D array of ephemeris times.
        """
        if not isinstance(et, np.ndarray):
            raise TypeError("et must be a numpy.ndarray")
        if et.ndim != 1:
            raise ValueError("et must be a 1D numpy array")
        self.et = et

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "AbsoluteDateArray":
        """
        Construct an AbsoluteDateArray object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the time information.
                The dictionary should contain the following key-value pairs:
                - "time_format" (str): The date-time format, either
                                       "Gregorian_Date" or "Julian_Date"
                                       (case-insensitive).
                - "time_scale" (str): The time scale, e.g., "UTC"
                                      (case-insensitive).
                                      See :class:`eosimutils.time.TimeScale` for options.

                For "Gregorian_Date" format:
                - "times" (list of str): List of date-times in YYYY-MM-DDTHH:MM:SS.SSS format.

                For "Julian_Date" format:
                - "times" (list of float): List of Julian Dates.

        Returns:
            AbsoluteDateArray: AbsoluteDateArray object.
        """
        time_scale: TimeScale = TimeScale.get(dict_in["time_scale"])
        time_format: TimeFormat = TimeFormat.get(dict_in["time_format"])

        # Load SPICE kernel files
        load_spice_kernels()

        ephemeris_times = []
        if time_scale == TimeScale.UTC:
            if time_format == TimeFormat.GREGORIAN_DATE:
                for calendar_date_str in dict_in["times"]:
                    ephemeris_times.append(spice.str2et(calendar_date_str))
            elif time_format == TimeFormat.JULIAN_DATE:
                for jd in dict_in["times"]:
                    time_string = f"jd {jd}"  # Format as Julian Date string
                    ephemeris_times.append(spice.str2et(time_string))
            else:
                raise ValueError(f"Unsupported date-time format: {time_format}")
        else:
            raise ValueError(f"Unsupported time scale: {time_scale}.")

        return cls(et=np.array(ephemeris_times))

    def to_astropy_time(self) -> Astropy_Time:
        """
        Convert the ephemeris times to an Astropy Time object (UTC scale).

        Returns:
            Astropy_Time: Astropy Time object representing the vector of times.
        """
        # Use AbsoluteDate.to_dict to retrieve the Gregorian UTC string for consistency.
        utc_strings = [
            AbsoluteDate(t).to_dict("GREGORIAN_DATE", "UTC")["calendar_date"]
            for t in self.et
        ]
        return Astropy_Time(utc_strings, scale="utc")

    def to_skyfield_time(self):
        """
        Convert the ephemeris times to a Skyfield Time object.

        Returns:
            skyfield.timelib.Time: Skyfield Time object representing the vector of times.
        """
        ts = Skyfield_Load.timescale()
        years, months, days, hours, minutes, seconds = [], [], [], [], [], []
        for t in self.et:
            # Get the UTC string using AbsoluteDate conversion for consistency.
            utc_string = AbsoluteDate(t).to_dict("GREGORIAN_DATE", "UTC")[
                "calendar_date"
            ]
            date_part, time_part = utc_string.split("T")
            y, m, d = map(int, date_part.split("-"))
            h, mi = map(int, time_part.split(":")[:2])
            s = int(float(time_part.split(":")[2]))
            years.append(y)
            months.append(m)
            days.append(d)
            hours.append(h)
            minutes.append(mi)
            seconds.append(s)
        return ts.utc(years, months, days, hours, minutes, seconds)

    def to_dict(
        self,
        time_format: Union[str, EnumBase] = "GREGORIAN_DATE",
        time_scale: Union[str, EnumBase] = "UTC",
    ) -> Dict[str, Any]:
        """
        Convert the AbsoluteDateArray object to a dictionary. For each ephemeris time,
        an ISO UTC string is generated (if Gregorian) or a Julian Date is computed.

        Args:
            time_format (str or EnumBase): The desired time format. Options are "GREGORIAN_DATE"
                                           or "JULIAN_DATE". Default is "GREGORIAN_DATE".
            time_scale (str or EnumBase): The time scale to use (e.g., "UTC"). Default is "UTC".

        Returns:
            dict: Dictionary with keys:
                - "time_format": the chosen format,
                - "times": a list of times (ISO strings/Gregorian or float/Julian Date),
                - "time_scale": the time scale.
        """
        # Convert each ephemeris time using AbsoluteDate.to_dict for consistency.
        times_list = []
        upper_format = str(time_format).upper()
        for t in self.et:
            ad_dict = AbsoluteDate(t).to_dict(time_format, time_scale)
            if upper_format == "GREGORIAN_DATE":
                times_list.append(ad_dict["calendar_date"])
            elif upper_format == "JULIAN_DATE":
                times_list.append(ad_dict["jd"])
            else:
                raise ValueError(f"Unsupported time_format: {time_format}")
        return {
            "time_format": str(time_format),
            "times": times_list,
            "time_scale": str(time_scale),
        }

"""
.. module:: eosimutils.time
    :synopsis: Collection of classes and functions for handling time information.

The time module provides classes and functions for representing, converting,
and manipulating time data.

**Internal Representation**:
The module maintains time internally in the SPICE Ephemeris Time (ET) format, which
corresponds to Barycentric Dynamical Time (TDB).

**Time Formats:**
Time formats define how time is represented. The module supports:
- **Gregorian Date**
- **Julian Date**

**Time Scales:**
Time scales define the method for measuring time. The module currently supports:
- **UTC (Coordinated Universal Time)**

**Key Features:**
- The module provides methods to convert between Gregorian Date, Julian Date (UTC time scale).
- It can convert from the eosimutils time objects to SPICE ET, Astropy and Skyfield time objects.
- The AbsoluteDateArray class allows efficient handling of multiple time points using NumPy arrays.

**Example Applications:**
- Defining mission/ orbit epochs.
- Defining orbit states at specific times.
- Defining array of time points for simulations.
- Converting between different time representations for analysis.
- Time intervals for representing observation windows or ground station contact windows.

**Constants:**
- `JD_OF_J2000 (float)`: Julian Date of the J2000 epoch (2451545.0).

**Example dictionary representations:**

AbsoluteDate
{
    "time_format": "Gregorian_Date",
    "calendar_date": "2017-07-14T19:46:00.0",
    "time_scale": "utc",
}

AbsoluteDateArray
{
    "time_format": "JULIAN_DATE",
    "jd": [2457949.323622, 2457949.3236278, 2457950.323622],
    "time_scale": "UTC"
}

AbsoluteDateIntervalArray
{
    "start_times": {
        "time_format": "GREGORIAN_DATE",
        "calendar_date": ["2025-03-10T14:30:00.000", "2025-03-11T14:30:00.000"],
        "time_scale": "UTC"
    },
    "stop_times": {
        "time_format": "GREGORIAN_DATE",
        "calendar_date": ["2025-03-10T15:30:00.000", "2025-03-11T15:30:00.000"],
        "time_scale": "UTC"
    }
}
"""

from typing import Dict, Any, Union, Tuple

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
                - "time_format" (str): The date-time format, e.g.,
                                       "Gregorian_Date" or "Julian_Date"
                                       (case-insensitive).
                                       See :class:`eosimutils.time.TimeFormat` for options.
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
                time_string = f"{jd} JDUTC"  # Format as Julian Date string
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

    def __add__(self, value):
        """Add a number of seconds to the AbsoluteDate object.

        Args:
            value (float): The number of seconds to add.

        Returns:
            AbsoluteDate: A new AbsoluteDate object with the updated time.
        """
        return AbsoluteDate(self.ephemeris_time + value)

    def __repr__(self):
        """Return a string representation of the AbsoluteDate."""
        return f"AbsoluteDate({self.ephemeris_time})"


class AbsoluteDateArray:
    """
    Vectorized representation of time in Ephemeris Time (ET).

    This class stores a set of time points as a 1D numpy array (in Ephemeris Time, ET)
    for efficiency. It provides vectorized methods to convert to other time representations,
    such as Astropy Time objects and Skyfield Time objects, as well as exporting time information
    to a dictionary.

    Attributes:
        ephemeris_time (np.ndarray): 1D numpy array of ephemeris times.
    """

    def __init__(self, ephemeris_time: np.ndarray) -> None:
        """
        Constructor for the AbsoluteDateArray class.

        Args:
            ephemeris_time (np.ndarray): 1D array of ephemeris times.
        """
        if not isinstance(ephemeris_time, np.ndarray):
            raise TypeError("ephemeris_time must be a numpy.ndarray")
        if ephemeris_time.ndim != 1:
            raise ValueError("ephemeris_time must be a 1D numpy array")
        self.ephemeris_time = ephemeris_time

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "AbsoluteDateArray":
        """
        Construct an AbsoluteDateArray object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the time information.
                The dictionary should contain the following key-value pairs:
                - "time_format" (str): The date-time format, e.g.,
                                       "Gregorian_Date" or "Julian_Date"
                                       (case-insensitive).
                                       See :class:`eosimutils.time.TimeFormat` for options.
                - "time_scale" (str): The time scale, e.g., "UTC"
                                      (case-insensitive).
                                      See :class:`eosimutils.time.TimeScale` for options.

                For "Gregorian_Date" format:
                - "calendar_date" (list of str): List of date-times in YYYY-MM-DDTHH:MM:SS.SSS
                format.

                For "Julian_Date" format:
                - "jd" (list of float): List of Julian Dates.

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
                for calendar_date_str in dict_in["calendar_date"]:
                    ephemeris_times.append(spice.str2et(calendar_date_str))
            elif time_format == TimeFormat.JULIAN_DATE:
                for jd in dict_in["jd"]:
                    time_string = f"{jd} JDUTC"  # Format as Julian Date string
                    ephemeris_times.append(spice.str2et(time_string))
            else:
                raise ValueError(f"Unsupported date-time format: {time_format}")
        else:
            raise ValueError(f"Unsupported time scale: {time_scale}.")

        return cls(ephemeris_time=np.array(ephemeris_times))

    def to_astropy_time(self) -> Astropy_Time:
        """
        Convert the ephemeris times to an Astropy Time object (UTC scale).

        Returns:
            Astropy_Time: Astropy Time object representing the vector of times.
        """
        # Use AbsoluteDate.to_dict to retrieve the Gregorian UTC string for consistency.
        utc_strings = [
            AbsoluteDate(t).to_dict("GREGORIAN_DATE", "UTC")["calendar_date"]
            for t in self.ephemeris_time
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
        for t in self.ephemeris_time:
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

    def to_spice_ephemeris_time(self) -> np.ndarray:
        """Convert the AbsoluteDateArray object to an array of SPICE Ephemeris Time(s) (ET).
        In the SPICE toolkit, ET Means TDB.

        Returns:
            np.ndarray: 1D numpy float array of Ephemeris Times (ET).
        """
        return self.ephemeris_time

    def to_dict(
        self,
        time_format: Union[str, EnumBase] = "GREGORIAN_DATE",
        time_scale: Union[str, EnumBase] = "UTC",
    ) -> Dict[str, Any]:
        """
        Convert the AbsoluteDateArray object to a dictionary.

        Args:
            time_format (str): The desired time format. Options are "GREGORIAN_DATE"
                                or "JULIAN_DATE". Default is "GREGORIAN_DATE".
            time_scale (str): The time scale to use (e.g., "UTC"). Default is "UTC".

        Returns:
            dict: Dictionary with keys:
                - "time_format": the chosen format,
                - "calendar_date" or "jd": list of times (ISO strings/Gregorian or float/jd),
                - "time_scale": the time scale.
        """
        # Convert each ephemeris time using AbsoluteDate.to_dict for consistency.
        times_list = []
        upper_format = str(time_format).upper()
        for t in self.ephemeris_time:
            ad_dict = AbsoluteDate(t).to_dict(time_format, time_scale)
            if upper_format == "GREGORIAN_DATE":
                times_list.append(ad_dict["calendar_date"])
            elif upper_format == "JULIAN_DATE":
                times_list.append(ad_dict["jd"])
            else:
                raise ValueError(f"Unsupported time_format: {time_format}")
        return {
            "time_format": str(time_format),
            (
                "calendar_date" if upper_format == "GREGORIAN_DATE" else "jd"
            ): times_list,
            "time_scale": str(time_scale),
        }

    def __len__(self):
        """Return the length of the AbsoluteDateArray."""
        return len(self.ephemeris_time)

    def __getitem__(self, index):
        """Get an item or a slice from the AbsoluteDateArray.

        Args:
            index (int or slice): Index or slice of the item(s) to retrieve.

        Returns:
            AbsoluteDate or AbsoluteDateArray: Selected item(s) as AbsoluteDate
                                                or AbsoluteDateArray.
        """
        if isinstance(index, slice):
            # Handle slicing
            return AbsoluteDateArray(self.ephemeris_time[index])
        else:
            # Handle single index
            return AbsoluteDate(self.ephemeris_time[index])

    def __eq__(self, value):
        """Check equality of two AbsoluteDateArray objects.

        Args:
            value (AbsoluteDateArray): The AbsoluteDateArray object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(value, AbsoluteDateArray):
            return False
        return np.array_equal(self.ephemeris_time, value.ephemeris_time)

    def __repr__(self):
        """Return a string representation of the AbsoluteDateArray."""
        return f"AbsoluteDateArray({self.ephemeris_time})"

class AbsoluteDateIntervalArray:
    """
    Representation of time intervals in Ephemeris Time (ET).

    This class stores a set of time intervals as two AbsoluteDateArray objects (start and stop times in Ephemeris Time, ET).
    It provides methods to convert to other time representations, such as Astropy Time objects
    and Skyfield Time objects, as well as importing/exporting time interval information from/to a dictionary.

    Attributes:
        start_times (AbsoluteDateArray): AbsoluteDateArray of start times in Ephemeris Time (ET).
        stop_times (AbsoluteDateArray): AbsoluteDateArray of stop times in Ephemeris Time (ET).
    """

    def __init__(self, start_times: AbsoluteDateArray, stop_times: AbsoluteDateArray) -> None:
        """
        Constructor for the AbsoluteDateIntervalArray class.

        Args:
            start_times (AbsoluteDateArray): AbsoluteDateArray of start times in Ephemeris Time (ET).
            stop_times (AbsoluteDateArray): AbsoluteDateArray of stop times in Ephemeris Time (ET).
        """
        if not isinstance(start_times, AbsoluteDateArray) or not isinstance(stop_times, AbsoluteDateArray):
            raise TypeError("start_times and stop_times must be AbsoluteDateArray objects")
        if len(start_times) != len(stop_times):
            raise ValueError("start_times and stop_times must have the same length")
        if not np.all(start_times.ephemeris_time <= stop_times.ephemeris_time):
            raise ValueError("Each start time must be less than or equal to its corresponding stop time")

        self.start_times = start_times
        self.stop_times = stop_times

    @classmethod
    def from_dict(cls, dict_in: Dict[str, Any]) -> "AbsoluteDateIntervalArray":
        """
        Construct an AbsoluteDateIntervalArray object from a dictionary.

        Args:
            dict_in (dict): Dictionary with the time interval information.
                The dictionary should contain the following key-value pairs:
                - "start_times": Dictionary for AbsoluteDateArray.from_dict.
                - "stop_times": Dictionary for AbsoluteDateArray.from_dict.

        Returns:
            AbsoluteDateIntervalArray: AbsoluteDateIntervalArray object.
        """
        start_times = AbsoluteDateArray.from_dict(dict_in["start_times"])
        stop_times = AbsoluteDateArray.from_dict(dict_in["stop_times"])
        return cls(start_times=start_times, stop_times=stop_times)

    def to_dict(
        self,
        time_format: Union[str, EnumBase] = "GREGORIAN_DATE",
        time_scale: Union[str, EnumBase] = "UTC",
    ) -> Dict[str, Any]:
        """
        Convert the AbsoluteDateIntervalArray object to a dictionary.

        Args:
            time_format (str): The desired time format. Options are "GREGORIAN_DATE"
                               or "JULIAN_DATE". Default is "GREGORIAN_DATE".
            time_scale (str): The time scale to use (e.g., "UTC"). Default is "UTC".

        Returns:
            dict: Dictionary with keys:
                - "start_times": Dictionary representation of start_times.
                - "stop_times": Dictionary representation of stop_times.
        """
        return {
            "start_times": self.start_times.to_dict(time_format, time_scale),
            "stop_times": self.stop_times.to_dict(time_format, time_scale),
        }

    def to_spice_ephemeris_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the AbsoluteDateIntervalArray object to SPICE Ephemeris Time (ET).

        Returns:
            tuple: A tuple containing two 1D numpy arrays:
                - First array: Start times in ET.
                - Second array: Stop times in ET.
        """
        return self.start_times.to_spice_ephemeris_time(), self.stop_times.to_spice_ephemeris_time()

    def __len__(self):
        """Return the number of intervals in the AbsoluteDateIntervalArray."""
        return len(self.start_times)

    def __getitem__(self, index):
        """Get an interval or a slice of intervals from the AbsoluteDateIntervalArray.

        Args:
            index (int or slice): Index or slice of the interval(s) to retrieve.

        Returns:
            AbsoluteDateIntervalArray: Selected interval(s) as AbsoluteDateIntervalArray.
        """
        if not isinstance(index, slice):
            index = slice(index, index + 1)  # Convert single index to slice
        return AbsoluteDateIntervalArray(
            AbsoluteDateArray(self.start_times.ephemeris_time[index]),
            AbsoluteDateArray(self.stop_times.ephemeris_time[index])
        )

    def __repr__(self):
        """Return a string representation of the AbsoluteDateIntervalArray."""
        return f"AbsoluteDateIntervalArray(start_times={self.start_times}, stop_times={self.stop_times})"
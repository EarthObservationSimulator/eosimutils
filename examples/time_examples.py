"""
This example demonstrates usage of the eosimutils.time module:
- Creating an AbsoluteDate from a dictionary.
- Converting AbsoluteDate to Gregorian and Julian formats.
- Converting AbsoluteDate to Astropy and Skyfield times.
- Using the AbsoluteDateArray class for vectorized time handling.
"""

from eosimutils.time import AbsoluteDate, AbsoluteDateArray
import numpy as np
from astropy.time import Time as AstropyTime

# Example 1: Create an AbsoluteDate from a Gregorian dictionary
gregorian_dict = {
    "time_format": "Gregorian_Date",
    "calendar_date": "2025-03-17T12:00:00.0",
    "time_scale": "utc",
}

abs_date = AbsoluteDate.from_dict(gregorian_dict)
print("Ephemeris Time (ET):", abs_date.ephemeris_time)

# Convert AbsoluteDate to Gregorian date dictionary output
gregorian_out = abs_date.to_dict("GREGORIAN_DATE", "UTC")
print("Gregorian Output:", gregorian_out)

# Convert AbsoluteDate to Julian date dictionary output
julian_out = abs_date.to_dict("Julian_Date", "UTC")
print("Julian Output:", julian_out)

# Convert to Astropy Time
astropy_time = abs_date.to_astropy_time()
print("Astropy Time:", astropy_time.iso)

# Convert to Skyfield Time
skyfield_time = abs_date.to_skyfield_time()
print("Skyfield Time:", skyfield_time.utc_strftime("%Y-%m-%d %H:%M:%S UTC"))

# Example 2: Using the AbsoluteDateArray class for vectorized time handling
t_values = abs_date.ephemeris_time + np.linspace(-100, 100, 5)
abs_dates_obj = AbsoluteDateArray.from_dict({
    "time_format": "Julian_Date",
    "jd": t_values.tolist(),
    "time_scale": "UTC"
})

# Convert AbsoluteDateArray object to an Astropy Time object
astropy_times_vector = abs_dates_obj.to_astropy_time()
print("Vectorized Astropy Times:")
print(astropy_times_vector.iso)

# Export AbsoluteDateArray object to a dictionary in Gregorian format
times_dict = abs_dates_obj.to_dict("GREGORIAN_DATE", "UTC")
print("AbsoluteDateArray Object to Dict (Gregorian):", times_dict)

# Export AbsoluteDateArray object to a dictionary in Julian format
times_dict_jd = abs_dates_obj.to_dict("Julian_Date", "UTC")
print("AbsoluteDateArray Object to Dict (Julian):", times_dict_jd)
"""
.. module:: eosimutils.timeseries
   :synopsis: Module for handling timeseries data.

The timeseries module provides functionality for handling time-varying data, 
represented as arrays associated with time points.

**Key features:**

- Support for scalar and vector data.
- Interpolation and resampling of data.
- Arithmetic operations between timeseries or with scalars.
- Handling of gaps (NaN values) in data.
- Serialization and deserialization to/from dictionaries.

**Example Applications:**
- Representing spacecraft ephemeris data.

**Example dictionary representations:**
- Timeseries
    {
        'time': {   'time_format': 'JULIAN_DATE',
                    'jd': [2451545.0, 2451546.0, 2451547.0, 2451548.0],
                    'time_scale': 'UTC'
                },
        'data': [   [   1,  2,  3,  4  ],
                    [
                        [4.0, 2.0, 0.5],
                        [1.0, nan, 5.0],
                        [0.0, 1.0, 2.0],
                        [2.0, 3.5, 5.0]
                    ]
                ],
        'headers': [    ['index'],
                        ['x', 'y', 'z']
                ],
        'interpolator': 'linear'
    }

"""

import numpy as np
from scipy.interpolate import interp1d
from .time import AbsoluteDateArray


def _group_contiguous(indices):
    """
    Group a list of indices into contiguous segments.

    This function takes a list of numerical indices (e.g., [0, 1, 3, 4, 7, 8])
    and groups them into sublists where each sublist contains consecutive indices.
    This is useful when interpolating over time-series data: by grouping
    valid (non-NaN) indices into contiguous segments, interpolation is applied only over
    stretches of valid data so that gaps (NaN regions) remain unfilled.

    Original Array:
    +---+---+-----+---+---+-----+-----+---+---+
    | 1 | 2 | NaN | 4 | 5 | NaN | NaN | 8 | 9 |
    +---+---+-----+---+---+-----+-----+---+---+
    0   1   2     3   4   5     6     7   8   <-- Indices
    Valid Indices: [0, 1, 3, 4, 7, 8]

    Valid Indices:
    +---+---+         +---+---+         +---+---+
    | 1 | 2 |         | 4 | 5 |         | 8 | 9 |
    +---+---+         +---+---+         +---+---+
    0   1             3   4             7   8   <-- Contiguous Groups
    Contiguous Groups: [ [0, 1], [3, 4], [7, 8] ]

    Args:
        indices (list of int): A list of numerical indices.

    Returns:
        list of numpy.ndarray: An array of arrays, each containing a continuous sequence of indices.
    """
    # Convert indices to a NumPy array for vectorized operations.
    indices = np.asarray(indices)
    if indices.size == 0:
        # No indices provided; return an empty list.
        return []
    # Identify positions where the difference is not 1 and split there.
    split_idx = np.where(np.diff(indices) != 1)[0] + 1
    groups = np.split(indices, split_idx)
    return groups


class Timeseries:
    """
    General class for representing time-varying data with support for multiple arrays.

    +-------------------+
    |   Timeseries      |
    +-------------------+
    |                   |
    |   time            |
    |   +-------------+ |
    |   | time (1D)   | |  --> [t1, t2, t3, ..., tn]  (AbsoluteDateArray)
    |   +-------------+ |
    |                   |
    |   data            | 'data' is a list containing the below numpy array(s).
    |   +-------------+ |
    |   | Array 1 (1D)| |  --> [d1, d2, d3, ..., dn]  (Scalar data)
    |   +-------------+ |
    |   | Array 2 (2D)| |  --> [[v11, v12, v13],      (Vector data)
    |   |             | |       [v21, v22, v23],      (Each row corresponds to a time)
    |   |             | |       ...,
    |   |             | |       [vn1, vn2, vn3]]
    |   +-------------+ |
    |                   |
    |   headers         | 'headers' is a list containing the below list(s) of headers.
    |   +-------------+ |
    |   | Header 1    | |  --> "true_anomaly" (for scalar data)
    |   +-------------+ |
    |   | Header 2    | |  --> ["x", "y", "z"] (for vector data)
    |   +-------------+ |
    |                   |
    +-------------------+

    Attributes:
        time (AbsoluteDateArray): Contains a 1D numpy array of AbsoluteDateArray time objects.
        data (list): List of numpy arrays. Each array can be 1D (scalar) or 2D (vector).
        headers (list): List of headers for the data arrays. For vectors, headers are nested lists.
    
        When performing arithmetic/boolean operations take note of the following:
        
        Boolean logical operations:
        - Both Timeseries must have the same time grid. This is because interpolation of a Boolean
        array onto a new time grid would produce an array of NaNs. Thus an ensuing logical operation
        would be on an array of NaNs which would give invalid results.
        - When performed on numeric values, non-zero values (including NaNs, negative values, etc.)
        are treated as True, while zero values are treated as False.

        Arithmetic operations:
        - If the 'other' Timeseries object has a different time-grid, it is interpolated onto the
        `self.time` time-grid.
        - Arithmetic operations with Boolean values will result in NaNs.


    """

    def __init__(
        self,
        time: "AbsoluteDateArray",
        data: list,
        headers: list = None,
        interpolator: str = "linear",
    ):
        """
        Initialize a Timeseries instance.

        Args:
            time (AbsoluteDateArray): Time values provided as an AbsoluteDateArray object.
            data (list): List of numpy arrays (can be numeric or boolean).
            headers (list, optional): List of headers for the data arrays. Defaults to None.
            interpolator (str, optional): Interpolation method. Defaults to "linear".

        Raises:
            TypeError: If `time` is not an AbsoluteDateArray object.
            ValueError: If data arrays do not match the length of `time`.
            ValueError: If the number of headers does not match the number of data arrays.
        """
        if not isinstance(time, AbsoluteDateArray):
            raise TypeError("time must be an AbsoluteDateArray object.")
        for arr in data:
            if arr.shape[0] != time.ephemeris_time.shape[0]:
                raise ValueError(
                    "Each data array must have the same number of rows as time."
                )
            if not np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
                raise TypeError("Data arrays must be numeric or boolean.")
        self.time = time

        self.data = data
        if headers is not None:
            if len(headers) != len(data):
                raise ValueError(
                    f"Expected {len(data)} headers but got {len(headers)}."
                )
            self.headers = headers
        else:
            self.headers = [
                (
                    [f"col_{i}_{j}" for j in range(arr.shape[1])]
                    if arr.ndim == 2
                    else f"col_{i}"
                )
                for i, arr in enumerate(data)
            ]
        self.interpolator = interpolator

    def _resample_data(self, new_time: np.ndarray, method: str = "linear") -> tuple[AbsoluteDateArray, list, list]:
        """
        Resample data arrays based on interpolation over contiguous segments.

        If the `new_time` has the same time grid as the `self.time`,
        the original Timeseries data is returned without modification.

        If a new time point falls outside a contiguous block of valid data, it remains NaN.

        Note that resampling of a boolean array does not involve interpolation; instead,
        the resulting array will contain NaNs for the new time points.

        Args:
            new_time (np.ndarray): New time points for resampling.
            method (str, optional): Interpolation method. Defaults to "linear".

        Returns:
            tuple: A tuple containing:
                - AbsoluteDateArray: Resampled time array.
                - list: Resampled data arrays.
                - list: Headers of the data arrays.
        """
        new_data = []
        original_time = (
            self.time.ephemeris_time
        )  # use underlying ephemeris times

        # if the original and new time arrays are identical
        # we can skip resampling
        if np.array_equal(original_time, new_time):
            return self.time, self.data, self.headers
        
        for arr in self.data:
            # If the data array is boolean, return an array of NaNs
            if np.issubdtype(arr.dtype, np.bool_):
                new_data.append(np.full(new_time.shape, np.nan))
                continue

            # If this data array is scalar
            if arr.ndim == 1:
                # isnan returns an array with the same dimensions
                # as the input, but with boolean values indicating NaN positions.
                # np.where returns a tuple of arrays, one for each dimension,
                # indicating the indices of the elements that satisfy the condition.
                # the [0] extracts the array of indices from the tuple in the 1d case.
                valid_indices = np.where(~np.isnan(arr))[0]

                # If less than 2 valid points, cannot interpolate.
                if valid_indices.size < 2:
                    new_arr = np.full(new_time.shape, np.nan)
                else:
                    new_arr = np.full(new_time.shape, np.nan)
                    # Changed to pass valid_indices directly.
                    groups = _group_contiguous(valid_indices)
                    for group in groups:
                        if group.size < 2:
                            continue
                        seg_time = original_time[group]
                        seg_data = arr[group]
                        mask = (new_time >= seg_time[0]) & (
                            new_time <= seg_time[-1]
                        )
                        interp_func = interp1d(
                            seg_time,
                            seg_data,
                            kind=method,
                            bounds_error=False,
                            fill_value=np.nan,
                        )
                        new_arr[mask] = interp_func(new_time[mask])
                new_data.append(new_arr)
            # If this data array is a vector
            else:
                # For each column in a vector, work independently.
                new_arr = np.full((len(new_time), arr.shape[1]), np.nan)
                for i in range(arr.shape[1]):
                    col = arr[:, i]
                    valid_indices = np.where(~np.isnan(col))[0]
                    if valid_indices.size < 2:
                        continue
                    # Changed to pass valid_indices directly.
                    groups = _group_contiguous(valid_indices)
                    for group in groups:
                        if group.size < 2:
                            continue
                        seg_time = original_time[group]
                        seg_data = col[group]
                        mask = (new_time >= seg_time[0]) & (
                            new_time <= seg_time[-1]
                        )
                        interp_func = interp1d(
                            seg_time,
                            seg_data,
                            kind=method,
                            bounds_error=False,
                            fill_value=np.nan,
                        )
                        new_arr[mask, i] = interp_func(new_time[mask])
                new_data.append(new_arr)
        return AbsoluteDateArray(new_time), new_data, self.headers

    def _remove_gaps_data(self):
        """
        Remove leading and trailing gaps (NaN values) from time and data arrays.

        This function identifies valid (non-NaN) data points and trims the time and data arrays
        to exclude any leading or trailing gaps.

        Returns:
            tuple: A tuple containing:
                - AbsoluteDateArray: Time array with gaps removed.
                - list: Data arrays with gaps removed.
                - list: Headers of the data arrays.
        """
        original_time = self.time.ephemeris_time
        if self.data[0].ndim == 1:
            valid = ~np.isnan(self.data[0])
        else:
            valid = ~np.isnan(self.data[0][:, 0])
        if not np.any(valid):
            return (
                AbsoluteDateArray(original_time[0:0]),
                [arr[0:0] for arr in self.data],
                self.headers,
            )
        first = np.argmax(valid)
        last = len(valid) - np.argmax(valid[::-1])
        return (
            AbsoluteDateArray(original_time[first:last]),
            [arr[first:last] for arr in self.data],
            self.headers,
        )

    def to_dict(self, time_format="GREGORIAN_DATE", time_scale="UTC"):
        """
        Serialize the Timeseries instance into a dictionary.

        Converts the AbsoluteDateArray time object via its own `to_dict` method,
        transforms each numpy data array to a native Python list for JSON compatibility,
        and retains the headers as provided.

        Args:
            time_format (str, optional): The format in which to serialize the time values.
                Defaults to "GREGORIAN_DATE".
            time_scale (str, optional): The time scale to use for serialization.
                Defaults to "UTC".

        Returns:
            dict: A dictionary representation of this Timeseries instance.
        """
        # Serialize the time attribute using AbsoluteDateArray.to_dict.
        serialized_time = self.time.to_dict(
            time_format=time_format, time_scale=time_scale
        )
        # Convert each numpy array in 'data' to a list using ndarray.tolist() for portability.
        # Boolean flags is converted to integers (0 and 1).
        serialized_data = [
            arr.tolist() if not np.issubdtype(arr.dtype, np.bool_) else arr.astype(int).tolist()
            for arr in self.data
        ]
        return {
            "time": serialized_time,
            "data": serialized_data,
            "headers": self.headers,
            "interpolator": self.interpolator,
        }

    @classmethod
    def from_dict(cls, dct):
        """
        Deserialize a dictionary into a new Timeseries instance.

        Reconstructs the AbsoluteDateArray object using its `from_dict` method,
        converts each nested list in the data back to a numpy array,
        and applies headers accordingly.

        If the 'data' has an array of 0s and 1s, it will be treated as a boolean array.

        Args:
            dct (dict): A dictionary containing the keys 'time', 'data', and optionally 'headers'.

        Returns:
            Timeseries: A new Timeseries object constructed from the dictionary.
        """
        # Reconstruct the 'time' component using AbsoluteDateArray.from_dict.
        time_instance = AbsoluteDateArray.from_dict(dct["time"])
        # Convert each list in 'data' back to a numpy array.
        reconstructed_data = [
            np.array(item, dtype=bool) if isinstance(item[0], int) and all(x in [0, 1] for x in item)
            else np.array(item)
            for item in dct["data"]
        ]
        # Retrieve headers if available; otherwise, use None.
        headers = dct.get("headers", None)
        # Return a new Timeseries object constructed with the deserialized attributes.
        instance = cls(time_instance, reconstructed_data, headers)
        instance.interpolator = dct.get("interpolator", "linear")
        return instance

    def _arithmetic_op(self, other, op):
        """
        Perform arithmetic (e.g., addition, subtraction) or logical operations between
        another Timeseries object or with a scalar.

        This method supports operations between two Timeseries objects or between a
        Timeseries and a scalar. When operating on two Timeseries, the `other` Timeseries
        is resampled onto the time grid of `self` (using the underlying ephemeris times)
        before performing the operation. (If both the Timeseries have the same time grid,
        the operation is performed directly without resampling.)

        Arithmetic operations on Boolean arrays results in an array of NaNs.

        Args:
            other (Timeseries or scalar): The operand for the operation.
            op (callable): The operation to perform (e.g., addition, subtraction, logical AND).

        Returns:
            Timeseries: A new Timeseries object with the result of the operation.

        Raises:
            TypeError: If the operand is neither a Timeseries nor a scalar.
        """
        if np.isscalar(other):
            # For scalar operations, the arithmetic naturally propagates NaNs.
            # Boolean arrays are replaced with NaNs.
            new_data = [
                op(arr, other) if not np.issubdtype(arr.dtype, np.bool_) else np.full_like(arr, np.nan, dtype=float)
                for arr in self.data
            ]
            return Timeseries(self.time, new_data, self.headers)
        elif isinstance(other, Timeseries):
            # Resample other onto self.time.ephemeris_time (using the underlying ephemeris times).
            other_resamp = other._resample_data(self.time.ephemeris_time)[1] #pylint: disable=protected-access
            # Perform vectorized operation for each data array.
            new_data = [
                op(arr, other_arr) if not np.issubdtype(arr.dtype, np.bool_) else np.full_like(arr, np.nan, dtype=float)
                for arr, other_arr in zip(self.data, other_resamp)
            ]
            return Timeseries(self.time, new_data, self.headers)
        else:
            raise TypeError("Operand must be a Timeseries or a scalar.")

    def __add__(self, other):
        """
        Adds another Timeseries or scalar to this Timeseries.

        Args:
            other (Timeseries or scalar): The operand for addition.

        Returns:
            Timeseries: A new Timeseries object with the result of the addition.
        """
        return self._arithmetic_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        """
        Subtracts another Timeseries or scalar from this Timeseries.

        Args:
            other (Timeseries or scalar): The operand for subtraction.

        Returns:
            Timeseries: A new Timeseries object with the result of the subtraction.
        """
        return self._arithmetic_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        """
        Multiplies this Timeseries by another Timeseries or scalar.

        Args:
            other (Timeseries or scalar): The operand for multiplication.

        Returns:
            Timeseries: A new Timeseries object with the result of the multiplication.
        """
        return self._arithmetic_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        """
        Divides this Timeseries by another Timeseries or scalar.

        Args:
            other (Timeseries or scalar): The operand for division.

        Returns:
            Timeseries: A new Timeseries object with the result of the division.
        """
        return self._arithmetic_op(other, lambda a, b: a / b)

    def logical_and(self, other):
        """
        Perform logical AND between this Timeseries and another Timeseries.
        If this operation is performed on a numeric value, any non-zero number—including
        negatives and NaNs—is treated as True, and the result of the operation is
        evaluated accordingly.

        Note that both the Timeseries must have the same time grid.

        Args:
            other (Timeseries or scalar): The operand for the logical AND.

        Returns:
            Timeseries: A new Timeseries object with the result of the logical AND.
        """
        if not np.array_equal(self.time, other.time):
            raise ValueError("Time grids do not match")
        # Perform vectorized operation for each data array.
        new_data = [
            np.logical_and(arr, other_arr)
            for arr, other_arr in zip(self.data, other.data)
        ]
        return Timeseries(self.time, new_data, self.headers)

    def logical_or(self, other):
        """
        Perform logical OR between this Timeseries and another Timeseries.
        If this operation is performed on a numeric value, any non-zero number—including
        negatives and NaNs—is treated as True, and the result of the operation is
        evaluated accordingly.

        Note that both the Timeseries must have the same time grid.

        Args:
            other (Timeseries or scalar): The operand for the logical OR.

        Returns:
            Timeseries: A new Timeseries object with the result of the logical OR.
        """
        if not np.array_equal(self.time, other.time):
            raise ValueError("Time grids do not match")
        # Perform vectorized operation for each data array.
        new_data = [
            np.logical_or(arr, other_arr)
            for arr, other_arr in zip(self.data, other.data)
        ]
        return Timeseries(self.time, new_data, self.headers)
    
    def logical_not(self):
        """
        Perform logical NOT on this Timeseries.
        If this operation is performed on a numeric value, any non-zero number—including
        negatives and NaNs—is treated as True, and the result of the operation is
        evaluated accordingly.

        Returns:
            Timeseries: A new Timeseries object with the result of the logical NOT.
        """
        new_data = [np.logical_not(arr) for arr in self.data]
        return Timeseries(self.time, new_data, self.headers)

"""Module for handling timeseries data."""

# pylint: disable=protected-access

import numpy as np
from scipy.interpolate import interp1d
from .time import AbsoluteDateArray


def _group_contiguous(indices):
    """
    Group a list of indices into contiguous segments.

    This function takes a list of numerical indices (e.g., [0, 1, 2, 4, 5, 7])
    and groups them into sublists where each sublist contains consecutive indices.
    This is useful when interpolating over time-series data: by grouping
    valid (non-NaN) indices into contiguous segments, interpolation is applied only over
    stretches of valid data so that gaps (NaN regions) remain unfilled.

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

    Attributes:
        time (AbsoluteDateArray): Contains a 1D numpy array of ephemeris times.
        data (list): List of numpy arrays. Each array can be 1D (scalar) or 2D (vector).
        headers (list): List of headers for the data arrays. For vectors, headers are nested lists.
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
            data (list): List of numpy arrays.
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
            if arr.shape[0] != time.et.shape[0]:
                raise ValueError(
                    "Each data array must have the same number of rows as time."
                )
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

    def _resample_data(self, new_time: np.ndarray, method: str = "linear"):
        """
        Resample data arrays based on interpolation over contiguous segments.

        If a new time point falls outside a contiguous block of valid data, it remains NaN.

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
        original_time = self.time.et  # use underlying ephemeris times
        for arr in self.data:
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
        original_time = self.time.et
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

    def to_dict(self):
        """
        Serialize the Timeseries instance into a dictionary.

        Converts the AbsoluteDateArray time object via its own `to_dict` method,
        transforms each numpy data array to a native Python list for JSON compatibility,
        and retains the headers as provided.

        Returns:
            dict: A dictionary representation of this Timeseries instance.
        """
        # Serialize the time attribute using AbsoluteDateArray.to_dict.
        serialized_time = self.time.to_dict()
        # Convert each numpy array in 'data' to a list using ndarray.tolist() for portability.
        serialized_data = [arr.tolist() for arr in self.data]
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

        Args:
            dct (dict): A dictionary containing the keys 'time', 'data', and optionally 'headers'.

        Returns:
            Timeseries: A new Timeseries object constructed from the dictionary.
        """
        # Reconstruct the 'time' component using AbsoluteDateArray.from_dict.
        time_instance = AbsoluteDateArray.from_dict(dct["time"])
        # Convert each list in 'data' back to a numpy array.
        reconstructed_data = [np.array(item) for item in dct["data"]]
        # Retrieve headers if available; otherwise, use None.
        headers = dct.get("headers", None)
        # Return a new Timeseries object constructed with the deserialized attributes.
        instance = cls(time_instance, reconstructed_data, headers)
        instance.interpolator = dct.get("interpolator", "linear")
        return instance

    def _arithmetic_op(self, other, op):
        """
        Perform arithmetic operations (e.g., addition, subtraction) between timeseries or
        with a scalar.

        Args:
            other (Timeseries or scalar): The operand for the operation.
            op (callable): The operation to perform (e.g., addition, subtraction).

        Returns:
            Timeseries: A new Timeseries object with the result of the operation.

        Raises:
            TypeError: If the operand is neither a Timeseries nor a scalar.
        """
        if np.isscalar(other):
            # For scalar operations, the arithmetic naturally propagates NaNs.
            new_data = [op(arr, other) for arr in self.data]
            return Timeseries(self.time, new_data, self.headers)
        elif isinstance(other, Timeseries):
            # Resample other onto self.time.et (using the underlying ephemeris times).
            other_resamp = other._resample_data(self.time.et)[1]
            # Perform vectorized operation for each data array.
            new_data = [
                op(arr, other_arr)
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

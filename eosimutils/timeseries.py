"""Module for handling timeseries data."""
import numpy as np
from scipy.interpolate import interp1d
from .time import AbsoluteDates

def _group_contiguous(indices):
    """
    Group a list of indices into contiguous segments.

    This function takes a list of numerical indices (e.g., [0, 1, 2, 4, 5, 7])
    and groups them into sublists where each sublist contains consecutive indices.
    This is useful when interpolating over time-series data: by grouping
    valid (non-NaN) indices into contiguous segments, interpolation is applied only over
    stretches of valid data so that gaps (NaN regions) remain unfilled.
    
    Parameters:
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
        time (AbsoluteDates): Contains a 1D numpy array of ephemeris times.
        data (list): List of numpy arrays. Each array can be 1D (scalar) or 2D (vector).
        headers (list): List of headers for the data arrays. For vectors, headers are nested lists.
    """
    def __init__(self, time: "AbsoluteDates", data: list,
                 headers: list = None, interpolator: str = "linear"):
        """
        Constructor for Timeseries.

        Args:
            time (AbsoluteDates): Time values provided as an AbsoluteDates object.
            data (list): List of numpy arrays.
            headers (list, optional): List of headers for the data arrays.
        """
        if not isinstance(time, AbsoluteDates):
            raise TypeError("time must be an AbsoluteDates object.")
        for arr in data:
            if arr.shape[0] != time.et.shape[0]:
                raise ValueError("Each data array must have the same number of rows as time.")
        self.time = time

        self.data = data
        if headers is not None:
            if len(headers) != len(data):
                raise ValueError(f"Expected {len(data)} headers but got {len(headers)}.")
            self.headers = headers
        else:
            self.headers = [
                [f"col_{i}_{j}" for j in range(arr.shape[1])] if arr.ndim == 2 else f"col_{i}"
                for i, arr in enumerate(data)
            ]
        self.interpolator = interpolator

    def _resample_data(self, new_time: np.ndarray, method: str = "linear"):
        """
        Returns new time and data arrays based on interpolation over contiguous segments.
        If a new time point falls outside a contiguous block of valid data, it remains NaN.
        """
        new_data = []
        original_time = self.time.et   # use underlying ephemeris times
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
                        mask = (new_time >= seg_time[0]) & (new_time <= seg_time[-1])
                        interp_func = interp1d(seg_time, seg_data, kind=method,
                                               bounds_error=False, fill_value=np.nan)
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
                        mask = (new_time >= seg_time[0]) & (new_time <= seg_time[-1])
                        interp_func = interp1d(seg_time, seg_data, kind=method,
                                               bounds_error=False, fill_value=np.nan)
                        new_arr[mask, i] = interp_func(new_time[mask])
                new_data.append(new_arr)
        return AbsoluteDates(new_time), new_data, self.headers

    def _remove_gaps_data(self):
        """
        Helper function that returns time and data arrays with leading and trailing
        gaps (samples with NaN in the first element) removed.
        """
        original_time = self.time.et
        if self.data[0].ndim == 1:
            valid = ~np.isnan(self.data[0])
        else:
            valid = ~np.isnan(self.data[0][:, 0])
        if not np.any(valid):
            return AbsoluteDates(original_time[0:0]), [arr[0:0] for arr in self.data], self.headers
        first = np.argmax(valid)
        last = len(valid) - np.argmax(valid[::-1])
        return AbsoluteDates(original_time[first:last]), \
            [arr[first:last] for arr in self.data], self.headers

    def to_dict(self):
        """
        Serialize the Timeseries instance into a dictionary.

        Converts the AbsoluteDates time object via its own to_dict method,
        transforms each numpy data array to a native Python list for JSON compatibility,
        and retains the headers as provided.
        
        Returns:
            dict: A dictionary representation of this Timeseries instance.
        """
        # Serialize the time attribute using AbsoluteDates.to_dict.
        serialized_time = self.time.to_dict()
        # Convert each numpy array in 'data' to a list using ndarray.tolist() for portability.
        serialized_data = [arr.tolist() for arr in self.data]
        return {
            "time": serialized_time,
            "data": serialized_data,
            "headers": self.headers,
            "interpolator": self.interpolator
        }

    @classmethod
    def from_dict(cls, dct):
        """
        Deserialize a dictionary into a new Timeseries instance.

        Reconstructs the AbsoluteDates object using its from_dict method,
        converts each nested list in the data back to a numpy array,
        and applies headers accordingly.
        
        Args:
            dct (dict): A dictionary containing the keys 'time', 'data', and optionally 'headers'.
        
        Returns:
            Timeseries: A new Timeseries object constructed from the dictionary.
        """
        # Reconstruct the 'time' component using AbsoluteDates.from_dict.
        time_instance = AbsoluteDates.from_dict(dct["time"])
        # Convert each list in 'data' back to a numpy array.
        reconstructed_data = [np.array(item) for item in dct["data"]]
        # Retrieve headers if available; otherwise, use None.
        headers = dct.get("headers", None)
        # Return a new Timeseries object constructed with the deserialized attributes.
        instance = cls(time_instance, reconstructed_data, headers)
        instance.interpolator = dct.get("interpolator", "linear")
        return instance

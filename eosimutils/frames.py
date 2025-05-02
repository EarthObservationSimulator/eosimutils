"""
.. module:: eosimutils.frames
   :synopsis: Reference frame representation..
"""

from .base import EnumBase
from scipy.spatial.transform import Rotation as R  # Import Rotation from scipy
from collections import deque
from typing import Callable, Dict, Union
import numpy as np

from eosimutils.time import AbsoluteDate, AbsoluteDateArray


class ReferenceFrame(EnumBase):
    """
    Enumeration of recognized Reference frames.

    Attributes:

        ICRF_EC (str): Earth centered inertial frame aligned to the ICRF
                        (International Celestial Reference Frame) .

                    The alignment of the ICRF is as defined in the SPICE toolkit.
                    This is implemented with the J2000 frame defined in the SPICE toolkit.
                    It seems that J2000 is same as ICRF.
                    In SPICE the center of any inertial frame is ALWAYS the solar system barycenter.
                    See Slide 12 and 7 in
                    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/17_frames_and_coordinate_systems.pdf

        ITRF (str): International Terrestrial Reference Frame.
                    This is implemented with the ITRF93 frame defined in the SPICE toolkit.

                    Also see:
                    https://rhodesmill.org/skyfield/api-framelib.html#skyfield.framelib.itrs

    """

    ICRF_EC = "ICRF_EC"  # Geocentric Celestial Reference Frame (ECI)
    ITRF = "ITRF"  # International Terrestrial Reference Frame (ECEF)
    # TEME = "TEME"  # True Equator Mean Equinox


# A TransformFunc maps a single AbsoluteDate to a scipy Rotation object
TransformFunc = Callable[[AbsoluteDate], R]

# A TransformsFunc maps an AbsoluteDateArray to a scipy Rotation object containing multiple rotations
TransformsFunc = Callable[[AbsoluteDateArray], R]


class FrameRegistry:
    """
    Registry for time-varying coordinate frame transformations.

    Underlying data structure:
    - Frames and transforms form a graph: nodes are frame names, edges are transform functions.
    - Adjacency list `_adj`: dict mapping each frame to a dict of neighbor-frame names → TransformFunc/TransformsFunc.
    - Querying A->B at time t:
    1. BFS to discover a path through intermediate frames.
    2. At each edge, call its TransformFunc(t) or TransformsFunc(ts) to get the rotation(s).
    3. Compose the rotations along the path to yield the composite rotation(s).

    Attributes:
        _adj (Dict[str, Dict[str, Dict[str, Union[TransformFunc, TransformsFunc]]]]):
            Adjacency list representing the frame graph.
            _adj[frame_i][frame_j] = {"single": f_{i->j}(t), "multiple": f_{i->j}(ts)}.
    """

    def __init__(self):
        """
        Initializes an empty adjacency list for the frame graph.
        """
        # Initialize empty adjacency list
        self._adj: Dict[
            str, Dict[str, Dict[str, Union[TransformFunc, TransformsFunc]]]
        ] = {}

    def add_transform(
        self,
        from_frame: str,
        to_frame: str,
        transform_func: TransformFunc,
        transforms_func: TransformsFunc = None,
        set_inverse: bool = True,
    ):
        """
        Registers a direct transform function from `from_frame` to `to_frame`.
        Optionally registers the inverse for the reverse direction.

        Args:
            from_frame (str): Name of the source frame.
            to_frame (str): Name of the target frame.
            transform_func (TransformFunc): Function f(t: AbsoluteDate) -> scipy Rotation object
                mapping from_frame→to_frame.
            transforms_func (TransformsFunc, optional): Function f(ts: AbsoluteDateArray) -> scipy
                Rotation object containing multiple rotations. Defaults to applying `transform_func`
                to each date in the array.
            set_inverse (bool, optional): Whether to automatically register the inverse transform.
                Default is True.
        """

        # Default transforms_func if not provided
        if transforms_func is None:

            def transforms_func(ts: AbsoluteDateArray):
                return R.from_quat(
                    [transform_func(t).as_quat() for t in ts.ephemeris_time]
                )

        # Build inverse functions for single and multiple dates
        def inv_func(t: AbsoluteDate, func=transform_func):
            return func(t).inv()

        def inv_transforms_func(ts: AbsoluteDateArray, func=transforms_func):
            return func(ts).inv()

        # Store forward transform in adjacency list
        if from_frame not in self._adj:
            self._adj[from_frame] = {}
        self._adj[from_frame][to_frame] = {
            "single": transform_func,
            "multiple": transforms_func,
        }

        # Store inverse transform if set_inverse is True
        if set_inverse:
            if to_frame not in self._adj:
                self._adj[to_frame] = {}
            self._adj[to_frame][from_frame] = {
                "single": inv_func,
                "multiple": inv_transforms_func,
            }

    def get_transform(
        self,
        from_frame: str,
        to_frame: str,
        t: Union[AbsoluteDate, AbsoluteDateArray],
    ) -> R:
        """
        Computes the composed transform from `from_frame` to `to_frame` at time `t`.

        Args:
            from_frame (str): Name of the source frame.
            to_frame (str): Name of the target frame.
            t (Union[AbsoluteDate, AbsoluteDateArray]): AbsoluteDate or AbsoluteDateArray
                representing the time(s).

        Returns:
            R: scipy Rotation object representing the composite transform(s).

        Raises:
            KeyError: If no path exists.
        """
        # Determine if input is single or multiple dates
        is_single = isinstance(t, AbsoluteDate)

        # Identity if same frame
        if from_frame == to_frame:
            return (
                R.identity() if is_single else R.identity(len(t.ephemeris_time))
            )

        visited = {from_frame}
        # Prepare initial BFS queue: (frame, accumulated_rotation)
        first_nbrs = self._adj.get(from_frame, {})
        queue = deque(
            [
                (
                    from_frame,
                    (
                        R.identity()
                        if is_single
                        else R.identity(len(t.ephemeris_time))
                    ),
                )
            ]
        )

        # BFS loop
        while queue:
            curr_frame, acc = queue.popleft()
            for nbr, funcs in self._adj.get(curr_frame, {}).items():
                if nbr in visited:
                    continue
                func = funcs["single"] if is_single else funcs["multiple"]
                rot = func(t)
                new_acc = rot * acc
                if nbr == to_frame:
                    return new_acc
                visited.add(nbr)
                queue.append((nbr, new_acc))

        # No path found
        raise KeyError(
            f"No transform path from '{from_frame}' to '{to_frame}' at time {t}"
        )


# Module-level singleton
registry = FrameRegistry()

# Add spice transforms

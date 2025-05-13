"""
.. module:: eosimutils.frame_registry
   :synopsis: Reference frame registry..
"""

from scipy.spatial.transform import Rotation as Scipy_Rotation
from collections import deque
from typing import Dict, Union
import numpy as np

from .frames import ReferenceFrame
from .time import AbsoluteDate, AbsoluteDateArray
from .attitude import Attitude, SpiceAttitude


class FrameRegistry:
    """
    Registry for time-varying coordinate frame transformations.

    Underlying data structure:
    - Frames and transforms form a graph: nodes are ReferenceFrames, edges are Attitudes.
    - Adjacency list `_adj`: dict maps each frame to a dict of neighbor ReferenceFrames → Attitude.
    - Querying A->B at time t:
        1. BFS to discover a path through intermediate frames.
        2. At each edge, call its Attitude.at(t) to get the rotation(s) and angular velocity(ies).
        3. Compose the rotations and angular velocities along the path to yield the transform.
    """

    def __init__(self):
        """
        Initializes an adjacency list for frame transformations.
        By default, it contains transforms for ICRF_EC and ITRF frames.
        """
        # Initialize empty adjacency list
        self._adj: Dict[ReferenceFrame, Dict[ReferenceFrame, Attitude]] = {}

        # Add spice transforms
        self.add_spice_transforms()

    def add_spice_transforms(self):
        """
        Adds SPICE transforms for ICRF_EC and ITRF frames.
        """
        # Add transforms from ICRF_EC to ITRF
        self.add_transform(
            SpiceAttitude(ReferenceFrame.ICRF_EC, ReferenceFrame.ITRF),
            False,
        )

        # Add transforms from ITRF to ICRF_EC
        self.add_transform(
            SpiceAttitude(ReferenceFrame.ITRF, ReferenceFrame.ICRF_EC),
            False,
        )

    def add_transform(
        self,
        attitude: Attitude,
        set_inverse: bool = True,
    ):
        """
        Registers a direct Attitude instance using its from_frame and to_frame attributes.
        Optionally registers the inverse for the reverse direction.

        Args:
            attitude (Attitude): Attitude instance mapping from_frame→to_frame.
            set_inverse (bool, optional): Whether to automatically register the inverse transform.
                Default is True.
        """
        from_frame = attitude.from_frame
        to_frame = attitude.to_frame

        # Store forward attitude in adjacency list
        if from_frame not in self._adj:
            self._adj[from_frame] = {}
        self._adj[from_frame][to_frame] = attitude

        # Store inverse attitude if set_inverse is True
        if set_inverse:
            if to_frame not in self._adj:
                self._adj[to_frame] = {}
            self._adj[to_frame][from_frame] = attitude.inverse()

    def get_transform(
        self,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
        t: Union[AbsoluteDate, AbsoluteDateArray],
    ) -> tuple[Scipy_Rotation, np.ndarray]:
        """
        Computes the composed transform from `from_frame` to `to_frame` at time `t`.

        Args:
            from_frame (ReferenceFrame): Source frame.
            to_frame (ReferenceFrame): Target frame.
            t (Union[AbsoluteDate, AbsoluteDateArray]): AbsoluteDate or AbsoluteDateArray
                representing the time(s).

        Returns:
            tuple[R, np.ndarray]: Scipy_Rotation object and angular velocity vector for transform.

        Raises:
            KeyError: If no path exists.
        """
        # Determine if input is single or multiple dates
        is_single = isinstance(t, AbsoluteDate)

        # Identity if same frame
        if from_frame == to_frame:
            return (
                (Scipy_Rotation.identity(), np.zeros(3))
                if is_single
                else (
                    Scipy_Rotation.identity(len(t.ephemeris_time)),
                    np.zeros((len(t.ephemeris_time), 3)),
                )
            )

        visited = {from_frame}
        zero_w = (
            np.zeros(3) if is_single else np.zeros((len(t.ephemeris_time), 3))
        )
        # Prepare initial BFS queue: (frame, accumulated_rotation, accumulated_angular_velocity)
        queue = deque(
            [
                (
                    from_frame,
                    (
                        Scipy_Rotation.identity()
                        if is_single
                        else Scipy_Rotation.identity(len(t.ephemeris_time))
                    ),
                    zero_w,
                )
            ]
        )

        # BFS loop
        while queue:
            curr_frame, acc_rot, acc_w = queue.popleft()
            for nbr, att in self._adj.get(curr_frame, {}).items():
                if nbr in visited:
                    continue
                rot, w = att.at(t)
                new_rot = rot * acc_rot
                new_w = rot.apply(acc_w) + w
                if nbr == to_frame:
                    return new_rot, new_w
                visited.add(nbr)
                queue.append((nbr, new_rot, new_w))

        # No path found
        raise KeyError(
            f"No transform path from '{from_frame}' to '{to_frame}' at time {t}"
        )

"""
.. module:: eosimutils.frame_registry
   :synopsis: Reference frame registry..
"""

from scipy.spatial.transform import Rotation as Scipy_Rotation
from collections import deque
from typing import Dict, Union, Any, List
import numpy as np

from .base import ReferenceFrame
from .time import AbsoluteDate, AbsoluteDateArray
from .orientation import Orientation, SpiceOrientation, ConstantOrientation


class FrameRegistry:
    """
    Registry for time-varying coordinate frame transformations.

    Underlying data structure:
    - Frames and transforms form a graph: nodes are ReferenceFrames and edges are Orientation
    instances.
    - Adjacency list `_adj`: maps each source ReferenceFrame to a list of Orientation edges.
    - Querying A->B at time t:
        1. BFS to discover a path through intermediate frames.
        2. At each edge, call its Orientation.at(t) to get the rotation(s)/angular velocity(ies).
        3. Compose the rotations and angular velocities along the path to yield the transform.
    """

    def __init__(self):
        """
        Initializes an adjacency list for frame transformations.
        By default, it contains transforms for ICRF_EC and ITRF frames.
        """
        # Initialize empty adjacency list
        self._adj: Dict[ReferenceFrame, List[Orientation]] = {}

        # Add spice transforms
        self.add_spice_transforms()

    def add_spice_transforms(self):
        """
        Adds SPICE transforms for ICRF_EC and ITRF frames.
        """

        # Add transforms from ICRF_EC to ITRF
        self.add_transform(
            SpiceOrientation(
                ReferenceFrame.get("ICRF_EC"), ReferenceFrame.get("ITRF")
            ),
            False,
        )
        # Add transforms from ITRF to ICRF_EC
        self.add_transform(
            SpiceOrientation(
                ReferenceFrame.get("ITRF"), ReferenceFrame.get("ICRF_EC")
            ),
            False,
        )

    def add_transform(
        self,
        orientation: Orientation,
        set_inverse: bool = True,
    ):
        """
        Registers a transformation between two reference frames using Orientation instance.

        This creates a directed edge in the graph from `from_frame` to `to_frame`, which
        are member variables of the Orientation instance.
        Optionally registers the inverse for the reverse direction.

        Args:
            orientation (Orientation): Orientation instance mapping from_frame→to_frame.
            set_inverse (bool, optional): Whether to automatically register the inverse transform.
                Default is True.
        """
        from_frame = orientation.from_frame
        # append forward edge
        self._adj.setdefault(from_frame, []).append(orientation)
        # optionally append inverse edge
        if set_inverse:
            inv = orientation.inverse()
            self._adj.setdefault(inv.from_frame, []).append(inv)

    def get_transform(
        self,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
        t: Union[AbsoluteDate, AbsoluteDateArray, None],
    ) -> tuple[Scipy_Rotation, np.ndarray]:
        """
        Computes the composed transform from `from_frame` to `to_frame` at time `t`.

        Args:
            from_frame (ReferenceFrame): Source frame.
            to_frame (ReferenceFrame): Target frame.
            t (Union[AbsoluteDate, AbsoluteDateArray, None]): AbsoluteDate, AbsoluteDateArray,
                or None representing the time(s).

        Returns:
            tuple[R, np.ndarray]: Scipy_Rotation object and angular velocity vector for transform.

        Raises:
            KeyError: If no path exists or if `t` is None and no ConstantOrientation-only path
                exists.
        """
        # Determine if input is single, multiple dates, or None
        is_single = isinstance(t, (AbsoluteDate, type(None)))

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
            for orient in self._adj.get(curr_frame, []):
                nbr = orient.to_frame
                if nbr in visited:
                    continue

                # If t is None, ensure all orientations are ConstantOrientation
                if t is None and not isinstance(orient, ConstantOrientation):
                    continue

                rot, w = orient.at(t)
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the FrameRegistry to a dictionary.

        The result contains a "transforms" key which maps to a list,
        where each entry is an orientation dict.

        Returns:
            dict: {"transforms": List[dict]} of all registered Orientation edges.
        """
        transforms = []
        for edges in self._adj.values():
            for orient in edges:
                transforms.append(orient.to_dict())
        return {"transforms": transforms}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameRegistry":
        """
        Deserialize a FrameRegistry from a dictionary.

        Args:
            data (dict): {"transforms": List[dict]} as produced by to_dict().

        Returns:
            FrameRegistry: New instance with each orientation added.
        """
        registry = cls()
        registry._adj.clear()
        for orient_data in data.get("transforms", []):
            orientation = Orientation.from_dict(orient_data)
            registry.add_transform(orientation, set_inverse=False)
        return registry

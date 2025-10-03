"""
.. module:: eosimutils.framegraph
   :synopsis: Registry for managing transformations between reference frames.

The `framegraph` module provides a graph-based registry for managing and querying
transformations between reference frames. It supports both orientation and position
transformations and enables seamless composition of transformations across multiple frames.

**Key Features**

Graph-Based Structure:
- Frames are represented as nodes, and transformations (orientations and/or positions)
    are represented as edges in the graph.
- Alternatively, could have used tree-structure to prevent multiple paths to compute the
    same transformation. Graph structure allows registering a chain of transformations
    (e.g., from frame A->B->C->D), but also directly registering the transformation
    from A->D for performance
- Directed edges for transformations, with optional automatic registration of
    inverse transformations.
- Transformation between `eosimutil` provided reference frames:
    `ICRF_EC` and `ITRF` are automatically registered.

Breadth-first search (BFS) algorithm for graph traversal:
- Computes transformations between frames at specific times by discovering paths
    through intermediate frames.
- Orientation and Position transformation are handled separately.
    Position transformations require corresponding orientation
    transformation to be available.

Orientation Transformations:
- Handles constant and time-varying orientations using `Orientation` subclasses:
    `ConstantOrientation` and `OrientationSeries`.
- Computes composed transformations between frames (at specific times)
    using breadth-first search (BFS).

Position Transformations:
- Supports translations between frame origins using `Cartesian3DPosition`
    or `PositionSeries`.
- Computes composed position transformations across multiple frames.

Serialization and Deserialization:
- Provides methods to serialize/deserialize the frame graph to/from a dictionary.

**Example Applications**
- Managing transformations between several connected reference frames.
- Interpolating time-varying transformations for visualization or analysis.

**Example graph illustration**

+-----------+       +-----------+
|  ICRF_EC  |------>|   LVLH    |
+-----------+       +-----------+
      ^                  ^
      |                  |
      v                  v
 +--------+      +------------------+
 |  ITRF  |      |  SC_BODY_FIXED   |
 +--------+      +------------------+
                         ^
                         |
                         v
              +-----------------------+
              |  SENSOR_BODY_FIXED    |
              +-----------------------+

An LVLH frame is defined relative to the ICRF_EC frame. The spacecraft body-fixed frame
is aligned with the LVLH frame, having zero offset. The sensor body-fixed frame is defined
relative to the spacecraft body-fixed frame, with a constant roll-axis offset for a
side-looking configuration. The double arrows indicate that bi-directional transformations
are registered.

**Example dictionary representation**
```
{
  "orientation_transforms": [
    {
      "orientation_type": "spice", "from": "ICRF_EC", "to": "ITRF"
    },
    {
      "orientation_type": "spice", "from": "ITRF", "to": "ICRF_EC"
    },
    {
      "orientation_type": "constant", "rotations": [0.0, 0.0, 1.57],
      "rotations_type": "EULER", "from": "A", "to": "B", "euler_order": "xyz"
    },
    {
      "orientation_type": "constant", "rotations": [0.0, 0.0, -1.57],
      "rotations_type": "EULER", "from": "A", "to": "D", "euler_order": "xyz"
    },
    ...
  ],
  "position_transforms": [
    {
      "from_frame": "ICRF_EC", "to_frame": "ITRF",
      "position": { "x": 0.0,  "y": 0.0, "z": 0.0,
        "frame": "ICRF_EC", "type": "Cartesian3DPosition"
      }
    },
    {
      "from_frame": "ITRF", "to_frame": "ICRF_EC",
      "position": { "x": 0.0,  "y": 0.0, "z": 0.0,
        "frame": "ITRF", "type": "Cartesian3DPosition"
      }
    },
    { "from_frame": "A", "to_frame": "B",
      "position": { "x": 1.0,  "y": 0.0, "z": 0.0,
        "frame": "A", "type": "Cartesian3DPosition"
      }
    },
    {
      "from_frame": "A",
      "to_frame": "D",
      "position": {"x": -0.0, "y": -4.0, "z": -0.0,
        "frame": "A", "type": "Cartesian3DPosition"
      }
    },
    {
      "from_frame": "A", "to_frame": "C",
      "position": { "x": 5.0, "y": 0.0, "z": 0.0,
        "frame": "A", "type": "Cartesian3DPosition"
      }
    },
    ...
  ]
}
```
"""

from scipy.spatial.transform import Rotation as Scipy_Rotation
from collections import deque
from typing import Dict, Union, Any, List
import numpy as np

from .base import ReferenceFrame
from .time import AbsoluteDate, AbsoluteDateArray
from .orientation import Orientation, SpiceOrientation, ConstantOrientation
from .state import Cartesian3DPosition
from .trajectory import PositionSeries


class FrameGraph:
    """
    Registry for time-varying coordinate frame transformations.

    Underlying data structure:
    - Frames and transforms form a graph: nodes are ReferenceFrames and edges are Orientation
    instances.
    - Adjacency list `_orientation_adj`: maps each source ReferenceFrame to a
        list of Orientation edges.
    - Querying A->B at time t:
        1. BFS to discover a path through intermediate frames.
        2. At each edge, call its Orientation.at(t) to get the rotation(s)/angular velocity(ies).
        3. Compose the rotations and angular velocities along the path to yield the transform.

    Likewise, position transforms are stored in a separate adjacency list `_pos_adj`:
    - Maps each source ReferenceFrame to a dictionary of target ReferenceFrames and their
        corresponding position transforms (either Cartesian3DPosition or PositionSeries).
    """

    def __init__(self):
        """
        Initializes an adjacency list for frame transformations.
        By default, it contains transforms for ICRF_EC and ITRF frames.
        """
        # Initialize empty adjacency list for orientation transforms
        self._orientation_adj: Dict[ReferenceFrame, List[Orientation]] = {}
        # Initialize empty adjacency list for position transforms
        self._pos_adj: Dict[
            ReferenceFrame,
            Dict[ReferenceFrame, Union[Cartesian3DPosition, PositionSeries]],
        ] = {}

        # Add spice transforms
        self.add_spice_transforms()

    def add_spice_transforms(self):
        """
        Adds SPICE transforms for ICRF_EC and ITRF frames.
        """

        # Add transforms from ICRF_EC to ITRF
        self.add_orientation_transform(
            SpiceOrientation(
                ReferenceFrame.get("ICRF_EC"), ReferenceFrame.get("ITRF")
            ),
            False,
        )
        # Add transforms from ITRF to ICRF_EC
        self.add_orientation_transform(
            SpiceOrientation(
                ReferenceFrame.get("ITRF"), ReferenceFrame.get("ICRF_EC")
            ),
            False,
        )

        # Add position transforms from ICRF_EC to ITRF
        self.add_pos_transform(
            ReferenceFrame.get("ICRF_EC"),
            ReferenceFrame.get("ITRF"),
            Cartesian3DPosition(0.0, 0.0, 0.0, ReferenceFrame.get("ICRF_EC")),
            False,
        )

        self.add_pos_transform(
            ReferenceFrame.get("ITRF"),
            ReferenceFrame.get("ICRF_EC"),
            Cartesian3DPosition(0.0, 0.0, 0.0, ReferenceFrame.get("ITRF")),
            False,
        )

    def add_orientation_transform(
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
        self._orientation_adj.setdefault(from_frame, []).append(orientation)
        # optionally append inverse edge
        if set_inverse:
            inv = orientation.inverse()
            self._orientation_adj.setdefault(inv.from_frame, []).append(inv)

    def add_pos_transform(
        self,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
        position: Union[Cartesian3DPosition, PositionSeries],
        set_inverse: bool = True,
    ):
        """
        Registers a translation between from_frame and to_frame.

        Args:
            from_frame (ReferenceFrame): Source frame.
            to_frame (ReferenceFrame): Target frame.
            position (Cartesian3DPosition or PositionSeries): Translation vector from
                `from_frame` origin to `to_frame` origin, expressed in from_frame coordinates.
            set_inverse (bool, optional): Whether to automatically register the inverse transform.
            If set to true, the orientation transformation from `from_frame` to `to_frame`
            must already be registered in the FrameGraph, or get_orientation_transform will
            raise an error.
        """
        if not isinstance(position, (Cartesian3DPosition, PositionSeries)):
            raise TypeError(
                "Position must be a Cartesian3DPosition or PositionSeries instance"
            )
        if from_frame is None or to_frame is None:
            raise ValueError("from_frame and/or to_frame cannot be None")
        if position.frame != from_frame:
            raise ValueError(
                "Position object must be expressed in from_frame coordinates."
            )
        if from_frame == to_frame:
            raise ValueError("from_frame and to_frame must be different")

        # Add transform to the adjacency list
        self._pos_adj.setdefault(from_frame, {})[to_frame] = position

        # Add inverse transform
        if set_inverse:
            if isinstance(position, Cartesian3DPosition):
                # Get orientation transformation from from_frame to to_frame
                rot, _ = self.get_orientation_transform(
                    from_frame, to_frame, None
                )
                # Get the position transformation v_inv from to_frame to from_frame,
                # expressed in to_frame coordinates
                v_inv = -rot.apply(position.to_numpy())
                inv_position = Cartesian3DPosition(
                    v_inv[0], v_inv[1], v_inv[2], to_frame
                )
            else:  # PositionSeries
                # Position as numpy array, shape (N, 3)
                v = position.data[0]
                # Get orientation transformation from from_frame to to_frame
                rot, _ = self.get_orientation_transform(
                    from_frame, to_frame, position.time
                )
                # Get the position transformation v_inv from to_frame to from_frame,
                # expressed in to_frame coordinates
                v_inv = -rot.apply(v)
                inv_position = PositionSeries(position.time, v_inv, to_frame)
            # Add inverse transform to adjacency list
            self._pos_adj.setdefault(to_frame, {})[from_frame] = inv_position

    def get_orientation_transform(
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
            for orient in self._orientation_adj.get(curr_frame, []):
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

    def get_pos_transform(
        self,
        from_frame: ReferenceFrame,
        to_frame: ReferenceFrame,
        t: Union[AbsoluteDate, AbsoluteDateArray, None],
    ) -> np.ndarray:
        """
        Computes the translation from `from_frame` origin to `to_frame` origin at time `t`.

        Args:
            from_frame (ReferenceFrame): Source frame.
            to_frame (ReferenceFrame): Target frame.
            t (AbsoluteDate, AbsoluteDateArray, or None): Time(s) of interest.

        Returns:
            np.ndarray: Array of shape (3,) or (N, 3) of position vectors in `from_frame`
            coordinates.

        Raises:
            KeyError: If no valid translation path exists.
        """
        # Determine single vs multiple times
        is_single = isinstance(t, (AbsoluteDate, type(None)))
        # Identity if same frame
        if from_frame == to_frame:
            return (
                np.zeros(3)
                if is_single
                else np.zeros((len(t.ephemeris_time), 3))
            )

        # BFS queue: (current_frame, pos_from_to_curr, rot_curr_to_from), where:
        # current_frame: ReferenceFrame of node being processed
        # pos_from_to_curr: np.ndarray from from_frame to current_frame (in from_frame coordinates)
        # rot_curr_to_from: Scipy_Rotation that transforms from current_frame to from_frame
        initial_offset = (
            np.zeros(3) if is_single else np.zeros((len(t.ephemeris_time), 3))
        )
        initial_rot = (
            Scipy_Rotation.identity()
            if is_single
            else Scipy_Rotation.identity(len(t.ephemeris_time))
        )
        queue = deque([(from_frame, initial_offset, initial_rot)])
        visited = set([from_frame])

        while queue:
            curr_frame, pos_from_to_curr, rot_curr_to_from = queue.popleft()
            # items() function returns an iterable of (tuple) key-value pairs
            # where key is the neighbor ReferenceFrame being processed and value is the
            # PositionSeries or Cartesian3DPosition
            for nbr_frame, pos_obj in self._pos_adj.get(curr_frame, {}).items():

                if nbr_frame in visited:
                    continue

                # If t is None, object cannot be a position series (which is time-varying)
                if t is None and isinstance(pos_obj, PositionSeries):
                    continue

                # Get the translation vector from curr_frame to nbr_frame, expressed in
                # curr_frame coordinates.
                if isinstance(pos_obj, Cartesian3DPosition):
                    pos_curr_to_nbr = pos_obj.to_numpy()
                else:
                    pos_curr_to_nbr = pos_obj.at(t)

                # Express pos_curr_to_nbr vector in from_frame coordinates:
                pos_curr_to_nbr = rot_curr_to_from.apply(pos_curr_to_nbr)
                pos_from_to_nbr = pos_from_to_curr + pos_curr_to_nbr

                # If we've reached the target frame, return the result
                if nbr_frame == to_frame:
                    return pos_from_to_nbr

                # Compute new accumulated rotation
                rot_nbr_to_curr, _ = self.get_orientation_transform(
                    nbr_frame, curr_frame, t
                )
                rot_nbr_to_from = rot_curr_to_from * rot_nbr_to_curr
                visited.add(nbr_frame)
                queue.append((nbr_frame, pos_from_to_nbr, rot_nbr_to_from))

        raise KeyError(
            f"No position path from '{from_frame}' to '{to_frame}' at time {t}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the FrameGraph to a dictionary.

        The result contains:
            - "orientation_transforms": a list of Orientation object dictionaries (see Orientation.to_dict())
            - "position_transforms": a list of position transforms, each as a dict with:
                - "from_frame": string name of the source ReferenceFrame,
                - "to_frame": string name of the target ReferenceFrame,
                - "position": dict representation of the position object (with a "type" key).

        Returns:
            dict: {
                "orientation_transforms": List[dict],      # orientation transforms
                "position_transforms": List[dict],  # position transforms
            }
        """
        transforms = []
        for edges in self._orientation_adj.values():
            for orient in edges:
                transforms.append(orient.to_dict())

        pos_transforms = []
        for from_frame, nbrs in self._pos_adj.items():
            for to_frame, pos_obj in nbrs.items():
                # Determine type and serialize accordingly
                pos_dict = pos_obj.to_dict()
                if pos_obj.__class__.__name__ == "Cartesian3DPosition":
                    pos_type = "Cartesian3DPosition"
                elif pos_obj.__class__.__name__ == "PositionSeries":
                    pos_type = "PositionSeries"
                else:
                    raise ValueError(
                        f"Unknown position type: {pos_obj.__class__.__name__}"
                    )
                pos_dict["type"] = pos_type
                pos_transforms.append(
                    {
                        "from_frame": from_frame.to_string(),
                        "to_frame": to_frame.to_string(),
                        "position": pos_dict,
                    }
                )

        return {
            "orientation_transforms": transforms,
            "position_transforms": pos_transforms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], set_inverse=False) -> "FrameGraph":
        """
        Deserialize a FrameGraph from a dictionary.

        Transformations between the base frames `ICRF_EC` and `ITRF` are automatically added.

        Args:
            data (dict): Dictionary with the following keys:
                - "orientation_transforms": List of orientation transform dicts (see Orientation.from_dict()).
                - "position_transforms": List of position transform dicts, each with:
                    - "from_frame": string name of the source ReferenceFrame,
                    - "to_frame": string name of the target ReferenceFrame,
                    - "position": dict with a "type" key and serialized position data.
                    
            set_inverse (bool, optional): Whether to automatically register inverse transforms.

        Returns:
            FrameGraph: New instance with all orientation and position transforms added.
        """
        registry = cls()
        registry._orientation_adj.clear()
        registry._pos_adj.clear()

        # Add orientation transforms
        for orient_data in data.get("orientation_transforms", []):
            orientation = Orientation.from_dict(orient_data)
            registry.add_orientation_transform(orientation, set_inverse=set_inverse)

        # Add position transforms
        for pos_data in data.get("position_transforms", []):
            from_frame = ReferenceFrame.get(pos_data["from_frame"])
            to_frame = ReferenceFrame.get(pos_data["to_frame"])
            if from_frame is None or to_frame is None:
                raise ValueError(
                    f"from_frame and/or to_frame not recognized. from_frame: {from_frame}, to_frame: {to_frame}"
                )
            pos_dict = pos_data["position"]
            pos_type = pos_dict.get("type")
            # Remove the type key before passing to from_dict
            pos_dict = {k: v for k, v in pos_dict.items() if k != "type"}
            if pos_type == "Cartesian3DPosition":
                position = Cartesian3DPosition.from_dict(pos_dict)
            elif pos_type == "PositionSeries":
                position = PositionSeries.from_dict(pos_dict)
            else:
                raise ValueError(f"Unknown position type: {pos_type}")
            registry.add_pos_transform(
                from_frame, to_frame, position, set_inverse=set_inverse
            )
        
        # Add spice transforms
        registry.add_spice_transforms()

        return registry

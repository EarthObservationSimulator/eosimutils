"""
.. module:: eosimutils.fieldofview
   :synopsis: Field-of-view (FOV) related classes and functions.

   Field-of-view (FOV) related classes and functions.
   Three main types of FOV are supported:
   (1) Circular, (2) Rectangular, and (3) Polygonal

   Each FOV type is associated with its own set of parameters and methods.
   The FieldOfViewFactory class is responsible for creating instances of the
   appropriate FOV class based on the provided specifications.

   References:
    - https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/getfov_c.html
    - https://naif.jpl.nasa.gov/naif/Ancillary_Data_Production_for_Cubesats_and_Lunar_Exploration_v2.pdf

"""

from typing import Type, Dict, Any, List, Union
import numpy as np

from .base import EnumBase
from .frames import ReferenceFrame


class FieldOfViewType(EnumBase):
    """Enumeration of supported FOV types (shapes)."""

    CIRCULAR = "CIRCULAR"
    RECTANGULAR = "RECTANGULAR"
    POLYGON = "POLYGON"


class FieldOfViewFactory:
    """Factory class to register and invoke the appropriate field-of-view class.

    This class allows registering FOV classes and retrieving instances
    of the appropriate FOV based on specifications.

    example:
        factory = FieldOfViewFactory()
        factory.register_fov("Circular", Circular)
        specs = {"type": "Circular", "diameter": 60}
        fov = factory.get_fov(specs)

    Attributes:
        _creators (Dict[str, Type]): A dictionary mapping field-of-view shape
                                     labels to their respective classes.
    """

    def __init__(self):
        """Initializes the FieldOfViewFactory and registers default FOVs."""
        self._creators: Dict[str, Type] = {}
        self.register_fov(FieldOfViewType.CIRCULAR.value, CircularFieldOfView)
        self.register_fov(
            FieldOfViewType.RECTANGULAR.value, RectangularFieldOfView
        )
        self.register_fov(FieldOfViewType.POLYGON.value, PolygonFieldOfView)

    def register_fov(self, fov_type: str, creator: Type) -> None:
        """Registers a FOV class with a specific type label.

        Args:
            fov_type (str): The label for the fov type.
            creator (Type): The fov class to register.
        """
        self._creators[fov_type] = creator

    def get_fov(self, specs: Dict[str, Any]) -> Any:
        """Retrieves an instance of the appropriate FOV based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing fov specifications.
                Must include a valid fov type in the "fov_type" key.

        Returns:
            Any: An instance of the appropriate fov class initialized
                 with the given specifications.

        Raises:
            KeyError: If the "fov_type" key is missing in the specifications dictionary.
            ValueError: If the specified fov type is not registered.
        """
        fov_type_str = specs.get("fov_type")
        if fov_type_str is None:
            raise KeyError(
                'FOV type key "fov_type" not found in specifications dictionary.'
            )

        if fov_type_str not in self._creators:
            raise ValueError(f'FOV type "{fov_type_str}" is not registered.')

        creator = self._creators[fov_type_str]
        return creator.from_dict(specs)


class CircularFieldOfView:
    """This class represents a circular field-of-view with a specified diameter.
    """

    def __init__(
        self, diameter: float, frame: Union[ReferenceFrame, str], boresight: Union[list, np.ndarray, None] = None
    ):
        """Initializes the CircularFieldOfView.

        Args:
            diameter (float): Angular diameter of the circular field-of-view in degrees.
            frame (Union[ReferenceFrame, str]): Reference frame for the field-of-view.
            boresight (Union[list, np.ndarray, None]): Boresight 3d-vector for the field-of-view.

        Raises:
            ValueError: If the diameter is not between 0 and 180 degrees.
        """
        if not (0 <= diameter <= 180):
            raise ValueError("diameter must be between 0 and 180 degrees.")
        self.diameter = float(diameter)
        self.frame = ReferenceFrame(frame)
        self.boresight = np.array(boresight if boresight is not None else [0.0, 0.0, 1.0])

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "CircularFieldOfView":
        """Creates a CircularFieldOfView object from a dictionary.

        Args:
            specs (Dict[str, Any]): Dictionary containing the specifications for the field-of-view.
                Expected keys:
                - "diameter" (float): Angular diameter of the circular field-of-view in degrees.
                - "frame" (str): Reference frame for the field-of-view.
                - "boresight" (list[float], optional): Boresight 3d-vector for the field-of-view.
                    Defaults to [0.0, 0.0, 1.0] (pointing in the +Z direction).

        Returns:
            CircularFieldOfView: An instance of the CircularFieldOfView class.
        """
        diameter = specs.get("diameter")
        frame = ReferenceFrame(specs.get("frame"))
        boresight = specs.get("boresight", [0.0, 0.0, 1.0])  # Default to +Z axis
        return cls(diameter, frame, boresight)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the CircularFieldOfView object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the CircularFieldOfView object.
        """
        return {
            "diameter": self.diameter,
            "frame": self.frame.value,
            "boresight": self.boresight.tolist(),
        }


class RectangularFieldOfView:
    """Represents a rectangular field-of-view (FOV) with specified parameters.
    """

    def __init__(
        self,
        frame: Union[ReferenceFrame, str],
        ref_vector: Union[list, np.ndarray],
        ref_angle: float,
        cross_angle: float,
        boresight: Union[list, np.ndarray, None] = None,
    ) -> None:
        """Initializes the RectangularFieldOfView object.

        Args:
            frame (Union[ReferenceFrame, str]): The reference frame in which the FOV is defined.
            ref_vector (Union[list, np.ndarray]): The reference 3d-vector defining the plane for the reference angle.
            ref_angle (float): Half of the total angular extent in the plane defined by the boresight and reference 3d-vector.
            cross_angle (float): Half of the total angular extent in the plane perpendicular to the reference 3d-vector.
            boresight (Union[list, np.ndarray, None]): The boresight 3d-vector of the FOV.
        """
        self.frame = ReferenceFrame(frame)
        self.ref_vector = np.array(ref_vector)
        self.ref_angle = ref_angle
        self.cross_angle = cross_angle
        self.boresight = np.array(boresight if boresight is not None else [0.0, 0.0, 1.0])

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "RectangularFieldOfView":
        """Creates a RectangularFieldOfView object from a dictionary.

        Args:
            specs (Dict[str, Any]): Dictionary containing the FOV specifications.
                Expected keys:
                - "frame" (str): The reference frame in which the FOV is defined.
                - "boresight" (Union[list, np.ndarray], optional): The boresight 3d-vector of the FOV.
                                                        Default is [0.0, 0.0, 1.0] (pointing in the +Z direction).
                - "ref_vector" (Union[list, np.ndarray]): The reference 3d-vector defining the plane for the reference angle.
                - "ref_angle" (float): Half of the total angular extent in the plane defined by the boresight and reference 3d-vector.
                - "cross_angle" (float): Half of the total angular extent in the plane perpendicular to the reference 3d-vector.

        Returns:
            RectangularFieldOfView: An instance of the RectangularFieldOfView class.

        Raises:
            ValueError: If ref_angle or cross_angle is not between 0 and 90 degrees.
        """
        ref_angle = specs["ref_angle"]
        cross_angle = specs["cross_angle"]

        if not (0 <= ref_angle <= 90):
            raise ValueError("ref_angle must be between 0 and 90 degrees.")
        if not (0 <= cross_angle <= 90):
            raise ValueError("cross_angle must be between 0 and 90 degrees.")

        return cls(
            frame=ReferenceFrame(specs["frame"]),
            boresight=specs.get("boresight", [0.0, 0.0, 1.0]),  # Default to +Z axis
            ref_vector=specs["ref_vector"],
            ref_angle=ref_angle,
            cross_angle=cross_angle,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the RectangularFieldOfView object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the RectangularFieldOfView object.
        """
        return {
            "frame": self.frame.value,
            "boresight": self.boresight.tolist(),
            "ref_vector": self.ref_vector.tolist(),
            "ref_angle": self.ref_angle,
            "cross_angle": self.cross_angle,
        }


class PolygonFieldOfView:
    """Represents a polygonal field-of-view (FOV) with specified parameters.
    """

    def __init__(
        self,
        frame: Union[ReferenceFrame, str],
        boundary_corners: List[Union[list, np.ndarray]],
        boresight: Union[list, np.ndarray, None] = None,
    ) -> None:
        """Initializes the PolygonFieldOfView object.

        Args:
            frame (Union[ReferenceFrame, str]): The reference frame in which the FOV is defined.
            boundary_corners (List[Union[list, np.ndarray]]): A list of vectors defining the corners of the FOV.
                                    The vectors should be listed in either clockwise or counterclockwise order.
            boresight (Union[list, np.ndarray, None]): The boresight 3d-vector of the FOV.
                                Defaults to [0.0, 0.0, 1.0] (pointing in the +Z direction).

        Raises:
            ValueError: If fewer than 3 vectors are provided in boundary_corners.
            If any vector in boundary_corners is not in the same hemisphere as the boresight vector.
            TypeError: If the frame is not of type ReferenceFrame or str.
        """
        if not isinstance(frame, (ReferenceFrame, str)):
            raise TypeError("Frame must be of type ReferenceFrame or str.")

        if len(boundary_corners) < 3:
            raise ValueError("At least 3 vectors must be defined in boundary_corners.")
                    
        self.frame = ReferenceFrame(frame)
        self.boundary_corners = [np.array(corner) for corner in boundary_corners]
        self.boresight = np.array(boresight if boresight is not None else [0.0, 0.0, 1.0])

        for corner in self.boundary_corners:
            if np.dot(corner, self.boresight) <= 0:
                raise ValueError("All boundary_corners must be in the same hemisphere as the boresight vector.")

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "PolygonFieldOfView":
        """Creates a PolygonFieldOfView object from a dictionary.

        Args:
            specs (Dict[str, Any]): Dictionary containing the FOV specifications.
                Expected keys:
                - "frame" (str): The reference frame.
                - "boresight" (Union[list, np.ndarray]): The boresight vector.
                        Default is [0.0, 0.0, 1.0] (pointing in the +Z direction).
                - "boundary_corners" (List[Union[list, np.ndarray]]): A list of vectors defining the corners of the FOV.
                    The vectors should be listed in either clockwise or counterclockwise order.

        Returns:
            PolygonFieldOfView: An instance of the PolygonFieldOfView class.
        """
        return cls(
            frame=ReferenceFrame(specs["frame"]),
            boresight=specs.get("boresight", [0.0, 0.0, 1.0]),  # Default to +Z axis
            boundary_corners=specs["boundary_corners"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the PolygonFieldOfView object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the PolygonFieldOfView object.
        """
        return {
            "frame": self.frame.value,
            "boresight": self.boresight.tolist(),
            "boundary_corners": [corner.tolist() for corner in self.boundary_corners],
        }

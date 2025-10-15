"""
.. module:: eosimutils.fieldofview
   :synopsis: Field-of-view (FOV) related classes and functions.

    The module provides classes and utilities for representing and managing different types of
    fields-of-view (FOV)
    Three main types of FOV are supported:
    (1) Circular, (2) Rectangular, and (3) Polygonal

    Each FOV type is associated with its own set of parameters and methods.
    The reference frame for each FOV needs to be specified.
    The FieldOfViewFactory class is responsible for creating instances of the
    appropriate FOV class based on the provided specifications.

    References:
        - https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/getfov_c.html
        - https://naif.jpl.nasa.gov/naif/Ancillary_Data_Production_for_Cubesats_and_Lunar_Exploration_v2.pdf # pylint: disable=line-too-long

    **Key Features**

    Field-of-View Representation:
    - **CircularFieldOfView**: Represents a circular FOV defined by its angular diameter, reference frame, and boresight vector.
    - **RectangularFieldOfView**: Represents a rectangular FOV defined by its reference frame, boresight vector, reference vector, and angular extents (reference angle and cross angle).
    - **PolygonFieldOfView**: Represents a polygonal FOV defined by its reference frame, boresight vector, and a list of boundary corner vectors.

    Factory Pattern:
    - **FieldOfViewFactory**: A factory class for creating instances of the appropriate FOV type based on a dictionary of specifications.
                             It supports dynamic registration of custom FOV types.

    **Example Applications**

    - Modeling the field-of-view of spacecraft sensors, ground-stations.
    - Defining custom FOV shapes for specialized instruments.

    **Example Dictionary Representations**

    **CircularFieldOfView**:
    ```python
    {
        "fov_type": "CIRCULAR",
        "diameter": 60.0,  # Angular diameter in degrees
        "frame": "ICRF_EC",  # Reference frame
        "boresight": [0.0, 0.0, 1.0]  # Optional boresight vector (default: +Z axis)
    }
    ```

    **RectangularFieldOfView**:
    ```python
    {
        "fov_type": "RECTANGULAR",
        "frame": "ICRF_EC",  # Reference frame
        "boresight": [0.0, 0.0, 1.0],  # Optional boresight vector (default: +Z axis)
        "ref_vector": [1.0, 0.0, 0.0],  # Reference vector defining the plane
        "ref_angle": 45.0,  # Half angular extent in the reference plane (degrees)
        "cross_angle": 30.0  # Half angular extent in the perpendicular plane (degrees)
    }
    ```

    **PolygonFieldOfView**:
    ```python
    {
        "fov_type": "POLYGON",
        "frame": "ICRF_EC",  # Reference frame
        "boresight": [0.0, 0.0, 1.0],  # Optional boresight vector (default: +Z axis)
        "boundary_corners": [  # List of vectors defining the polygon corners
            [0.5, 0.5, 0.707],
            [0.1, 0.2, 0.979],
            [0.3, 0.4, 0.866],
            [0.6, 0.0, 0.8]
        ]
    }
    ```

---

### **Error Handling**
- **CircularFieldOfView**: Ensures the diameter is between 0 and 180 degrees.
- **RectangularFieldOfView**: Validates that reference and cross angles are between 0 and 90 degrees.
- **PolygonFieldOfView**: Ensures at least three boundary corners are provided and that all corners lie in the same hemisphere as the boresight vector.
- **OmnidirectionalFieldOfView**: No specific parameters; represents an all-encompassing FOV.
- **FieldOfViewFactory**: Raises errors for missing or unregistered FOV types.

---

### **Example Usage**
```python
from eosimutils.fieldofview import FieldOfViewFactory

# Create a factory and retrieve a circular FOV
factory = FieldOfViewFactory()
specs = {
    "fov_type": "CIRCULAR",
    "diameter": 60.0,
    "frame": "ICRF_EC",
    "boresight": [0.0, 0.0, 1.0]
}
circular_fov = factory.from_dict(specs)
print(circular_fov.to_dict())
```

"""

from typing import Type, Dict, Any, List, Union, Callable
import numpy as np

from .base import EnumBase, ReferenceFrame


class FieldOfViewType(EnumBase):
    """Enumeration of supported FOV types (shapes)."""

    CIRCULAR = "CIRCULAR"
    RECTANGULAR = "RECTANGULAR"
    POLYGON = "POLYGON"
    OMNIDIRECTIONAL = "OMNIDIRECTIONAL"


class FieldOfViewFactory:
    """Factory class to register and create field-of-view (FOV) objects."""

    # Class-level registry for FOV types
    _registry: Dict[str, Type] = {}

    @classmethod
    def register_type(cls, fov_type: str) -> Callable[[Type], Type]:
        """
        Decorator to register an FOV class under a specific type name.

        Args:
            fov_type (str): The label for the FOV type.

        Returns:
            Callable[[Type], Type]: A decorator that registers the FOV class.
        """

        def decorator(fov_class: Type) -> Type:
            cls._registry[fov_type] = fov_class
            return fov_class

        return decorator

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> object:
        """
        Retrieves an instance of the appropriate FOV class based on specifications.

        Args:
            specs (Dict[str, Any]): A dictionary containing FOV specifications.
                Must include a valid FOV type in the "fov_type" key.

        Returns:
            object: An instance of the appropriate FOV class initialized with
                    the given specifications.

        Raises:
            KeyError: If the "fov_type" key is missing in the specifications dictionary.
            ValueError: If the specified FOV type is not registered.
        """
        fov_type_str = specs.get("fov_type")
        if fov_type_str is None:
            raise KeyError(
                'FOV type key "fov_type" not found in specifications dictionary.'
            )
        fov_class = cls._registry.get(fov_type_str)
        if not fov_class:
            raise ValueError(f'FOV type "{fov_type_str}" is not registered.')
        return fov_class.from_dict(specs)

@FieldOfViewFactory.register_type("OMNIDIRECTIONAL")
class OmnidirectionalFieldOfView:
    """This class represents an omnidirectional field-of-view (FOV) that covers the entire sphere."""

    def __init__(self):
        """Initializes the OmnidirectionalFieldOfView."""
        pass

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "OmnidirectionalFieldOfView":
        """Creates an OmnidirectionalFieldOfView object from a dictionary.

        Args:
            specs (Dict[str, Any]): Dictionary containing the specifications for the field-of-view.
                This class does not require any specific parameters.

        Returns:
            OmnidirectionalFieldOfView: An instance of the OmnidirectionalFieldOfView class.
        """
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the OmnidirectionalFieldOfView object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the OmnidirectionalFieldOfView object.
        """
        return {"fov_type": FieldOfViewType.OMNIDIRECTIONAL.value}

@FieldOfViewFactory.register_type("CIRCULAR")
class CircularFieldOfView:
    """This class represents a circular field-of-view with a specified diameter."""

    def __init__(
        self,
        diameter: float,
        frame: Union[ReferenceFrame, str],
        boresight: Union[list, np.ndarray, None] = None,
    ):
        """Initializes the CircularFieldOfView.

        Args:
            diameter (float): Angular diameter of the circular field-of-view in degrees.
            frame (Union[ReferenceFrame, str]): Reference frame for the field-of-view.
            boresight (Union[list, np.ndarray, None]): Boresight 3d-vector for the field-of-view.

        Raises:
            ValueError: If the diameter is not between 0 and 180 degrees.
        """
        if not 0 <= diameter <= 180:
            raise ValueError("diameter must be between 0 and 180 degrees.")
        self.diameter = float(diameter)
        self.frame = ReferenceFrame.get(frame)
        self.boresight = np.array(
            boresight if boresight is not None else [0.0, 0.0, 1.0]
        )

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
        boresight = specs.get(
            "boresight", [0.0, 0.0, 1.0]
        )  # Default to +Z axis
        return cls(diameter, frame, boresight)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the CircularFieldOfView object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the CircularFieldOfView object.
        """
        return {
            "fov_type": FieldOfViewType.CIRCULAR.value,
            "diameter": self.diameter,
            "frame": self.frame.to_string(),
            "boresight": self.boresight.tolist(),
        }

@FieldOfViewFactory.register_type("RECTANGULAR")
class RectangularFieldOfView:
    """Represents a rectangular field-of-view (FOV) with specified parameters."""

    def __init__(
        self,
        frame: Union[ReferenceFrame, str],
        ref_angle: float,
        cross_angle: float,
        ref_vector: Union[list, np.ndarray, None] = None,
        boresight: Union[list, np.ndarray, None] = None,
    ) -> None:
        """Initializes the RectangularFieldOfView object.

        Args:
            frame (Union[ReferenceFrame, str]): The reference frame in which the FOV is defined.
            ref_angle (float): Half of the total angular extent in the plane defined by the
                               boresight and reference 3d-vector.
            cross_angle (float): Half of the total angular extent in the plane perpendicular to
                                the reference 3d-vector.
            ref_vector (Union[list, np.ndarray, None]): The reference 3d-vector defining the plane
                                            for the reference angle. Defaults to [1.0, 0.0, 0.0]
                                            (pointing in the +X direction).
                                            (Default value is assigned inside the function.)
            boresight (Union[list, np.ndarray, None]): The boresight 3d-vector of the FOV.
                                Defaults to [0.0, 0.0, 1.0] (pointing in the +Z direction).
                                (Default value is assigned inside the function.)
        """
        self.frame = ReferenceFrame.get(frame)
        self.ref_vector = np.array(
            ref_vector if ref_vector is not None else [1, 0, 0]
        )
        self.ref_angle = ref_angle
        self.cross_angle = cross_angle
        self.boresight = np.array(
            boresight if boresight is not None else [0.0, 0.0, 1.0]
        )

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "RectangularFieldOfView":
        """Creates a RectangularFieldOfView object from a dictionary.

        Args:
            specs (Dict[str, Any]): Dictionary containing the FOV specifications.
                Expected keys:
                - "frame" (str): The reference frame in which the FOV is defined.
                - "boresight" (Union[list, np.ndarray], optional): The boresight 3d-vector
                                                of the FOV. Default is [0.0, 0.0, 1.0]
                                                (pointing in the +Z direction).
                - "ref_vector" (Union[list, np.ndarray], optional): The reference 3d-vector defining
                                                        the plane for the reference angle.
                                                        Default is [1.0, 0.0, 0.0]
                                                        (pointing in the +X direction).
                - "ref_angle" (float): Half of the total angular extent in the plane defined
                                        by the boresight and reference 3d-vector.
                - "cross_angle" (float): Half of the total angular extent in the plane
                                        perpendicular to the reference 3d-vector.

        Returns:
            RectangularFieldOfView: An instance of the RectangularFieldOfView class.

        Raises:
            ValueError: If ref_angle or cross_angle is not between 0 and 90 degrees.
        """
        ref_angle = specs["ref_angle"]
        cross_angle = specs["cross_angle"]

        if not 0 <= ref_angle <= 90:
            raise ValueError("ref_angle must be between 0 and 90 degrees.")
        if not 0 <= cross_angle <= 90:
            raise ValueError("cross_angle must be between 0 and 90 degrees.")

        return cls(
            frame=ReferenceFrame(specs["frame"]),
            boresight=specs.get(
                "boresight", [0.0, 0.0, 1.0]
            ),  # Default to +Z axis
            ref_vector=specs.get(
                "ref_vector", [1.0, 0.0, 0.0]
            ),  # Default to +X axis
            ref_angle=ref_angle,
            cross_angle=cross_angle,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the RectangularFieldOfView object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the RectangularFieldOfView object.
        """
        return {
            "fov_type": FieldOfViewType.RECTANGULAR.value,
            "frame": self.frame.to_string(),
            "boresight": self.boresight.tolist(),
            "ref_vector": self.ref_vector.tolist(),
            "ref_angle": self.ref_angle,
            "cross_angle": self.cross_angle,
        }

@FieldOfViewFactory.register_type("POLYGON")
class PolygonFieldOfView:
    """Represents a polygonal field-of-view (FOV) with specified parameters."""

    def __init__(
        self,
        frame: Union[ReferenceFrame, str],
        boundary_corners: List[Union[list, np.ndarray]],
        boresight: Union[list, np.ndarray, None] = None,
    ) -> None:
        """Initializes the PolygonFieldOfView object.

        Args:
            frame (Union[ReferenceFrame, str]): The reference frame in which the FOV is defined.
            boundary_corners (List[Union[list, np.ndarray]]): A list of vectors defining the
                                    corners of the FOV. The vectors should be listed in either
                                    clockwise or counterclockwise order.
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
            raise ValueError(
                "At least 3 vectors must be defined in boundary_corners."
            )

        self.frame = ReferenceFrame.get(frame)
        self.boundary_corners = [
            np.array(corner) for corner in boundary_corners
        ]
        self.boresight = np.array(
            boresight if boresight is not None else [0.0, 0.0, 1.0]
        )

        for corner in self.boundary_corners:
            if np.dot(corner, self.boresight) <= 0:
                raise ValueError(
                    "All boundary_corners must be in the same hemisphere as the boresight vector."
                )

    @classmethod
    def from_dict(cls, specs: Dict[str, Any]) -> "PolygonFieldOfView":
        """Creates a PolygonFieldOfView object from a dictionary.

        Args:
            specs (Dict[str, Any]): Dictionary containing the FOV specifications.
                Expected keys:
                - "frame" (str): The reference frame.
                - "boresight" (Union[list, np.ndarray]): The boresight vector.
                        Default is [0.0, 0.0, 1.0] (pointing in the +Z direction).
                - "boundary_corners" (List[Union[list, np.ndarray]]): A list of vectors
                            defining the corners of the FOV. The vectors should be listed
                            in either clockwise or counterclockwise order.

        Returns:
            PolygonFieldOfView: An instance of the PolygonFieldOfView class.
        """
        return cls(
            frame=ReferenceFrame.get(specs["frame"]),
            boresight=specs.get(
                "boresight", [0.0, 0.0, 1.0]
            ),  # Default to +Z axis
            boundary_corners=specs["boundary_corners"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the PolygonFieldOfView object to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the PolygonFieldOfView object.
        """
        return {
            "fov_type": FieldOfViewType.POLYGON.value,
            "frame": self.frame.to_string(),
            "boresight": self.boresight.tolist(),
            "boundary_corners": [
                corner.tolist() for corner in self.boundary_corners
            ],
        }

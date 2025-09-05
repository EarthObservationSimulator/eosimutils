"""
.. module:: eosimutils.base
   :synopsis: Collection of basic utility classes and functions.

Collection of basic utility classes and functions.
"""

from enum import Enum
import json


class EnumBase(str, Enum):
    """Enumeration of recognized types.
    All enum values defined by the inheriting class are
    expected to be in uppercase."""

    @classmethod
    def get(cls, key):
        """Attempts to parse a type from a string, otherwise returns None."""
        if isinstance(key, cls):
            return key
        elif isinstance(key, list):
            return [cls.get(e) for e in key]
        else:
            try:
                return cls(key.upper())
            except:  # pylint: disable=bare-except
                return None

    def to_string(self) -> str:
        """Returns the string representation of the enum value.

        Returns:
            str: The string representation of the enum value.
        """
        return str(self.value)

    def __eq__(self, other) -> bool:
        """Check equality between EnumBase and another object.

        Args:
            other (Any): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if isinstance(other, EnumBase):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other.upper()
        return False


class JsonSerializer:
    """Class for handling JSON serialization and deserialization."""

    @staticmethod
    def load_from_json(other_cls, file_path):
        """Load an object from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return other_cls.from_dict(data)

    @staticmethod
    def save_to_json(obj, file_path):
        """Save the object to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj.to_dict(), f, indent=4)


class SurfaceType(EnumBase):
    """
    Enumeration of recognized planetary surface types.

    Attributes:
        WGS84 (str): Represents the WGS84 ellipsoidal surface.
        SPHERE (str): Represents a spherical surface.
        NONE (str): Represents no surface.
    """

    WGS84 = "WGS84"
    SPHERE = "SPHERE"
    NONE = "NONE"


class RotationsType(EnumBase):
    """
    Enumeration of recognized rotation types.

    Attributes:
        QUATERNION (str): Represents quaternion rotations.
        EULER (str): Represents Euler angle rotations.
    """

    QUATERNION = "QUATERNION"
    EULER = "EULER"


class ReferenceFrame:
    """Reference frame registry with static and dynamic support.

    The ReferenceFrame class is a registry for managing reference frames.
    It allows for the creation, retrieval, and management of reference frames
    globally across the application.

    Reference frames are stored in a (class-variable) dictionary providing lookup by name.
    Hence, there can be only one ReferenceFrame registry shared globally across the application.

    **Key Features:**
    - Global Registry: Reference frames are stored in a class-level dictionary (`_registry`),
                        enabling global access and lookup by name.
    - Dynamic Addition: New reference frames can be dynamically added using the `add` method.
    - Pre-Registered Frames: `eosimutils` builtin frames (`ICRF_EC` and `ITRF`) are pre-registered.
    - String Representation: Frames can be converted to strings using the `to_string` method
                                or via `__str__` method.
    - Equality and Hashing: Frames can be compared using `__eq__` and are hashable via `__hash__`.
    - Retrieval: Frames can be retrieved by name using the `get` method. The `values` method
                returns all registered frames, while the `names` method provides a list of their names.

    **Description of the pre-registered frames:**
    ICRF_EC:    Earth centered inertial frame aligned to the ICRF (International Celestial Reference Frame) .

                The alignment of the ICRF is as defined in the SPICE toolkit.
                This is implemented with the J2000 frame defined in the SPICE toolkit.
                It seems that J2000 is same as ICRF.
                In SPICE the center of any inertial frame is ALWAYS the solar system barycenter.
                See Slide 12 and 7 in
                https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/17_frames_and_coordinate_systems.pdf #pylint: disable=line-too-long

    ITRF: International Terrestrial Reference Frame.
                This is implemented with the ITRF93 frame defined in the SPICE toolkit.

                Also see:
                https://rhodesmill.org/skyfield/api-framelib.html#skyfield.framelib.itrs

    """

    _registry = {}

    def __init__(self, name: str):
        self._name = name.upper()

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"ReferenceFrame('{self._name}')"

    def to_string(self):
        """
        Converts the reference frame to a string.

        Returns:
            str: The name of the reference frame.
        """
        return self._name

    def __eq__(self, other):
        if isinstance(other, ReferenceFrame):
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other.upper()

        return False

    def __hash__(self):
        return hash(self._name)

    @classmethod
    def add(cls, name: str):
        """
        Dynamically adds a new ReferenceFrame.

        Args:
            name (str): The name of the reference frame to add.

        Returns:
            ReferenceFrame: The newly added reference frame.

        Raises:
            ValueError: If the reference frame already exists.
        """
        name = name.upper()
        if name in cls._registry:
            raise ValueError(f"Frame '{name}' already exists.")
        cls._registry[name] = ReferenceFrame(name)
        return cls._registry[name]

    @classmethod
    def get(cls, name: str):
        """
        Retrieves a ReferenceFrame by name or returns the instance if already a ReferenceFrame.

        Args:
            name (str or ReferenceFrame): The name of the reference frame to retrieve,
                                            or a ReferenceFrame instance.

        Returns:
            ReferenceFrame or None: The reference frame if found, the instance
                                    if already a ReferenceFrame, or None if not found.
        """
        if isinstance(name, ReferenceFrame):
            return name
        name = name.upper()
        if name not in cls._registry:
            return None

        return cls._registry[name]

    @classmethod
    def values(cls):
        """
        Retrieves all registered reference frames.

        Returns:
            List[ReferenceFrame]: A list of all registered reference frames.
        """
        return list(cls._registry.values())

    @classmethod
    def names(cls):
        """
        Retrieves the names of all registered reference frames.

        Returns:
            List[str]: A list of all registered reference frame names.
        """
        return list(cls._registry.keys())

    @classmethod
    def delete(cls, name: str):
        """
        Deletes a ReferenceFrame from the registry.

        Args:
            name (str): The name of the reference frame to delete.

        Raises:
            ValueError: If the reference frame does not exist.
        """
        name = name.upper()
        if name not in cls._registry:
            raise ValueError(f"Frame '{name}' does not exist.")
        del cls._registry[name]


# Pre-register static frames
# pylint: disable=protected-access
ReferenceFrame._registry["ICRF_EC"] = ReferenceFrame("ICRF_EC")
# pylint: disable=protected-access
ReferenceFrame._registry["ITRF"] = ReferenceFrame("ITRF")

# See: https://en.wikipedia.org/wiki/World_Geodetic_System
WGS84_EARTH_EQUATORIAL_RADIUS = 6378.1370
WGS84_EARTH_FLATTENING = 1.0 / 298.257223563
WGS84_EARTH_POLAR_RADIUS = WGS84_EARTH_EQUATORIAL_RADIUS * (
    1.0 - WGS84_EARTH_FLATTENING
)

# See: https://en.wikipedia.org/wiki/Earth_radius
SPHERICAL_EARTH_MEAN_RADIUS = 6371.0087714

SUN_RADIUS = 695700.0  # Sun radius in km

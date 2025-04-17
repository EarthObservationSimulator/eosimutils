"""
.. module:: eosimutils.base
   :synopsis: Collection of basic utility classes and functions.

Collection of basic utility classes and functions.
"""

from enum import Enum


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

    ICRF_EC = "ICRF_EC"  # Earth centered inertial frame aligned to the ICRF (ECI)
    ITRF = "ITRF"  # International Terrestrial Reference Frame (ECEF)
    # TEME = "TEME"  # True Equator Mean Equinox

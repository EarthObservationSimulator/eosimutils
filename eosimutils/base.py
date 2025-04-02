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
        GCRF (str): Geocentric Celestial Reference Frame. See:
                    https://rhodesmill.org/skyfield/api-position.html#geocentric-position-relative-to-the-earth

                    In the context of the SPICE toolkit:
                    It seems that J2000 is same as ICRF. See Slide 12 in
                    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/17_frames_and_coordinate_systems.pdf
                    Further, GCRF is aligned to ICRF with the center at the center of the Earth.

        ITRF (str): International Terrestrial Reference Frame. See:
                    https://rhodesmill.org/skyfield/api-framelib.html#skyfield.framelib.itrs

    """

    GCRF = "GCRF"  # Geocentric Celestial Reference Frame (ECI)
    ITRF = "ITRF"  # International Terrestrial Reference Frame (ECEF)
    # TEME = "TEME"  # True Equator Mean Equinox

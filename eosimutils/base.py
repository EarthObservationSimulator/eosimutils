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


class RotationsType(EnumBase):
    """
    Enumeration of recognized rotation types.

    Attributes:
        QUATERNION (str): Represents quaternion rotations.
        EULER (str): Represents Euler angle rotations.
    """

    QUATERNION = "QUATERNION"
    EULER = "EULER"

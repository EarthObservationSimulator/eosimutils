"""
.. module:: eosimutils.frames
   :synopsis: Reference frame representation..
"""

from typing import Dict, Union, List, Optional


class ReferenceFrameMeta(type):
    """
    Metaclass for ReferenceFrame to allow iteration over class (enum-like behavior).
    """

    def __iter__(cls):
        return iter(cls._frames.values())


class ReferenceFrame(metaclass=ReferenceFrameMeta):
    """
    Enum-like class with support for dynamically added reference frames.

    Behaves like an Enum:
    - Allows class-level access to static members (e.g. ReferenceFrame.ICRF_EC)
    - Supports ReferenceFrame.get("ICRF_EC")
    - Supports iteration: for frame in ReferenceFrame
    - Supports dynamic addition: ReferenceFrame.add("NEW_FRAME")
    """

    _frames: Dict[str, "ReferenceFrame"] = {}

    def __init__(self, name: str):
        """
        Initializes a reference frame with a given name.

        Args:
            name (str): Name of the reference frame.
        """
        self._name = name.upper()
        ReferenceFrame._frames[self._name] = self

    def __str__(self):
        return self._name

    def __repr__(self):
        return f"ReferenceFrame('{self._name}')"

    def to_string(self) -> str:
        return self._name

    def __eq__(self, other: Union[str, "ReferenceFrame"]) -> bool:
        """
        Compares this reference frame with another.

        Args:
            other (Union[str, ReferenceFrame]): Other reference frame or string to compare with.

        Returns:
                bool: True if they are equal, False otherwise.
        """
        if isinstance(other, ReferenceFrame):
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other.upper()
        return False

    def __hash__(self):
        return hash(self._name)

    @classmethod
    def get(
        cls,
        key: Union[str, "ReferenceFrame", List[Union[str, "ReferenceFrame"]]],
    ) -> Optional[Union["ReferenceFrame", List["ReferenceFrame"]]]:
        """
        Retrieves a reference frame by name or instance.

        Args:
            key (Union[str, ReferenceFrame, List[Union[str, ReferenceFrame]]]): The name or
                instance of the reference frame.

        Returns:
            Optional[Union[ReferenceFrame, List[ReferenceFrame]]]: The reference frame instance
                or a list of instances.
        """
        if isinstance(key, list):
            return [cls.get(k) for k in key]
        if isinstance(key, ReferenceFrame):
            return key
        if isinstance(key, str):
            return cls._frames.get(key.upper())
        return None

    @classmethod
    def add(cls, name: str) -> "ReferenceFrame":
        """
        Dynamically add a new reference frame.

        Args:
            name (str): Name of the new reference frame.

        Returns:
            ReferenceFrame: The newly added reference frame.
        """
        name_upper = name.upper()
        if name_upper in cls._frames:
            raise ValueError(f"Frame '{name}' already exists.")
        instance = cls(name_upper)
        cls._frames[name_upper] = instance
        setattr(
            cls, name_upper, instance
        )  # Dynamically create a class-level attribute
        return instance

    @classmethod
    def values(cls) -> List["ReferenceFrame"]:
        return list(cls._frames.values())

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._frames.keys())


# Add enum-like static members
for _name in ["ICRF_EC", "ITRF"]:
    inst = ReferenceFrame(_name)
    setattr(ReferenceFrame, _name, inst)

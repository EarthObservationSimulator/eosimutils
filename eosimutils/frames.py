"""
.. module:: eosimutils.frames
   :synopsis: Reference frame representation..
"""


class ReferenceFrame:
    """Reference frame registry with static and dynamic support.

    Reference frames are stored in a (class-variable) dictionary providing lookup by name.
    Hence, there is only one ReferenceFrame registry shared globally across the application.
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
            name (str or ReferenceFrame): The name of the reference frame to retrieve, or a ReferenceFrame instance.

        Returns:
            ReferenceFrame or None: The reference frame if found, the instance if already a ReferenceFrame, or None if not found.
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


# Pre-register static frames
# pylint: disable=protected-access
ReferenceFrame._registry["ICRF_EC"] = ReferenceFrame("ICRF_EC")
# pylint: disable=protected-access
ReferenceFrame._registry["ITRF"] = ReferenceFrame("ITRF")

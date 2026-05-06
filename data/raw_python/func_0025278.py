def get_property_as_float(self, name: str) -> float:
        """Return the value of a float property.

        :return: The property value (float).

        Raises exception if property with name doesn't exist.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        return float(self.__instrument.get_property(name))
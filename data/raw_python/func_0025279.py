def set_property_as_float(self, name: str, value: float) -> None:
        """Set the value of a float property.

        :param name: The name of the property (string).
        :param value: The property value (float).

        Raises exception if property with name doesn't exist.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        self.__instrument.set_property(name, float(value))
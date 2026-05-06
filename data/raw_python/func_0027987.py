def GetName(self, number):
    """Retrieves the name of an enumeration value by number.

    Args:
      number (int): number.

    Returns:
      str: name of the enumeration value or None if no corresponding
          enumeration value was found.
    """
    value = self._data_type_definition.values_per_number.get(number, None)
    if not value:
      return None

    return value.name
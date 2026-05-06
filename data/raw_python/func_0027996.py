def AddValue(self, name, number, aliases=None, description=None):
    """Adds an enumeration value.

    Args:
      name (str): name.
      number (int): number.
      aliases (Optional[list[str]]): aliases.
      description (Optional[str]): description.

    Raises:
      KeyError: if the enumeration value already exists.
    """
    if name in self.values_per_name:
      raise KeyError('Value with name: {0:s} already exists.'.format(name))

    if number in self.values_per_number:
      raise KeyError('Value with number: {0!s} already exists.'.format(number))

    for alias in aliases or []:
      if alias in self.values_per_alias:
        raise KeyError('Value with alias: {0:s} already exists.'.format(alias))

    enumeration_value = EnumerationValue(
        name, number, aliases=aliases, description=description)

    self.values.append(enumeration_value)
    self.values_per_name[name] = enumeration_value
    self.values_per_number[number] = enumeration_value

    for alias in aliases or []:
      self.values_per_alias[alias] = enumeration_value
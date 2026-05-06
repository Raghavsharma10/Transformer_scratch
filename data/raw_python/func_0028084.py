def RegisterDefinition(self, data_type_definition):
    """Registers a data type definition.

    The data type definitions are identified based on their lower case name.

    Args:
      data_type_definition (DataTypeDefinition): data type definitions.

    Raises:
      KeyError: if data type definition is already set for the corresponding
          name.
    """
    name_lower = data_type_definition.name.lower()
    if name_lower in self._definitions:
      raise KeyError('Definition already set for name: {0:s}.'.format(
          data_type_definition.name))

    if data_type_definition.name in self._aliases:
      raise KeyError('Alias already set for name: {0:s}.'.format(
          data_type_definition.name))

    for alias in data_type_definition.aliases:
      if alias in self._aliases:
        raise KeyError('Alias already set for name: {0:s}.'.format(alias))

    self._definitions[name_lower] = data_type_definition

    for alias in data_type_definition.aliases:
      self._aliases[alias] = name_lower

    if data_type_definition.TYPE_INDICATOR == definitions.TYPE_INDICATOR_FORMAT:
      self._format_definitions.append(name_lower)
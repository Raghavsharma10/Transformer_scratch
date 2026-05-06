def DeregisterDefinition(self, data_type_definition):
    """Deregisters a data type definition.

    The data type definitions are identified based on their lower case name.

    Args:
      data_type_definition (DataTypeDefinition): data type definition.

    Raises:
      KeyError: if a data type definition is not set for the corresponding
          name.
    """
    name = data_type_definition.name.lower()
    if name not in self._definitions:
      raise KeyError('Definition not set for name: {0:s}.'.format(
          data_type_definition.name))

    del self._definitions[name]
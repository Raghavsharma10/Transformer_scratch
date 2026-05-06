def _ReadDefinition(self, definitions_registry, definition_values):
    """Reads a data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.

    Returns:
      DataTypeDefinition: data type definition or None.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if not definition_values:
      error_message = 'missing definition values'
      raise errors.DefinitionReaderError(None, error_message)

    name = definition_values.get('name', None)
    if not name:
      error_message = 'missing name'
      raise errors.DefinitionReaderError(None, error_message)

    type_indicator = definition_values.get('type', None)
    if not type_indicator:
      error_message = 'invalid definition missing type'
      raise errors.DefinitionReaderError(name, error_message)

    data_type_callback = self._DATA_TYPE_CALLBACKS.get(type_indicator, None)
    if data_type_callback:
      data_type_callback = getattr(self, data_type_callback, None)
    if not data_type_callback:
      error_message = 'unuspported data type definition: {0:s}.'.format(
          type_indicator)
      raise errors.DefinitionReaderError(name, error_message)

    return data_type_callback(definitions_registry, definition_values, name)
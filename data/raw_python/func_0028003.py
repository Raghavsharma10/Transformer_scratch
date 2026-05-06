def _ReadDataTypeDefinition(
      self, definitions_registry, definition_values, data_type_definition_class,
      definition_name, supported_definition_values):
    """Reads a data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      data_type_definition_class (str): data type definition class.
      definition_name (str): name of the definition.
      supported_definition_values (set[str]): names of the supported definition
          values.

    Returns:
      DataTypeDefinition: data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    aliases = definition_values.get('aliases', None)
    description = definition_values.get('description', None)
    urls = definition_values.get('urls', None)

    unsupported_definition_values = set(definition_values.keys()).difference(
        supported_definition_values)
    if unsupported_definition_values:
      error_message = 'unsupported definition values: {0:s}'.format(
          ', '.join(unsupported_definition_values))
      raise errors.DefinitionReaderError(definition_name, error_message)

    return data_type_definition_class(
        definition_name, aliases=aliases, description=description, urls=urls)
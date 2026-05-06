def _ReadStringDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a string data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      StringDefinition: string data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if is_member:
      supported_definition_values = (
          self._SUPPORTED_DEFINITION_VALUES_STRING_MEMBER)
    else:
      supported_definition_values = self._SUPPORTED_DEFINITION_VALUES_STRING

    definition_object = self._ReadElementSequenceDataTypeDefinition(
        definitions_registry, definition_values, data_types.StringDefinition,
        definition_name, supported_definition_values)

    encoding = definition_values.get('encoding', None)
    if not encoding:
      error_message = 'missing encoding'
      raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object.encoding = encoding

    return definition_object
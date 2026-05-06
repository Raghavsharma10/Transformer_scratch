def _ReadStreamDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a stream data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      StreamDefinition: stream data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if is_member:
      supported_definition_values = (
          self._SUPPORTED_DEFINITION_VALUES_ELEMENTS_MEMBER_DATA_TYPE)
    else:
      supported_definition_values = (
          self._SUPPORTED_DEFINITION_VALUES_ELEMENTS_DATA_TYPE)

    return self._ReadElementSequenceDataTypeDefinition(
        definitions_registry, definition_values, data_types.StreamDefinition,
        definition_name, supported_definition_values)
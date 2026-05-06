def _ReadConstantDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a constant data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      ConstantDataTypeDefinition: constant data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if is_member:
      error_message = 'data type not supported as member'
      raise errors.DefinitionReaderError(definition_name, error_message)

    value = definition_values.get('value', None)
    if value is None:
      error_message = 'missing value'
      raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object = self._ReadSemanticDataTypeDefinition(
        definitions_registry, definition_values, data_types.ConstantDefinition,
        definition_name, self._SUPPORTED_DEFINITION_VALUES_CONSTANT)
    definition_object.value = value

    return definition_object
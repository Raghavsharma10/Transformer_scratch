def _ReadPaddingDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a padding data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      PaddingtDefinition: padding definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if not is_member:
      error_message = 'data type only supported as member'
      raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object = self._ReadDataTypeDefinition(
        definitions_registry, definition_values, data_types.PaddingDefinition,
        definition_name, self._SUPPORTED_DEFINITION_VALUES_PADDING)

    alignment_size = definition_values.get('alignment_size', None)
    if not alignment_size:
      error_message = 'missing alignment_size'
      raise errors.DefinitionReaderError(definition_name, error_message)

    try:
      int(alignment_size)
    except ValueError:
      error_message = 'unuspported alignment size attribute: {0!s}'.format(
          alignment_size)
      raise errors.DefinitionReaderError(definition_name, error_message)

    if alignment_size not in (2, 4, 8, 16):
      error_message = 'unuspported alignment size value: {0!s}'.format(
          alignment_size)
      raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object.alignment_size = alignment_size

    return definition_object
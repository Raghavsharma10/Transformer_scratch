def _ReadIntegerDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads an integer data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      IntegerDataTypeDefinition: integer data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    definition_object = self._ReadFixedSizeDataTypeDefinition(
        definitions_registry, definition_values,
        data_types.IntegerDefinition, definition_name,
        self._SUPPORTED_ATTRIBUTES_INTEGER, is_member=is_member,
        supported_size_values=(1, 2, 4, 8))

    attributes = definition_values.get('attributes', None)
    if attributes:
      format_attribute = attributes.get('format', definitions.FORMAT_SIGNED)
      if format_attribute not in self._INTEGER_FORMAT_ATTRIBUTES:
        error_message = 'unsupported format attribute: {0!s}'.format(
            format_attribute)
        raise errors.DefinitionReaderError(definition_name, error_message)

      definition_object.format = format_attribute

    return definition_object
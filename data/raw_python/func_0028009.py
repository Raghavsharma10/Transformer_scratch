def _ReadFormatDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a format data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      FormatDefinition: format definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if is_member:
      error_message = 'data type not supported as member'
      raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object = self._ReadLayoutDataTypeDefinition(
        definitions_registry, definition_values, data_types.FormatDefinition,
        definition_name, self._SUPPORTED_DEFINITION_VALUES_FORMAT)

    # TODO: disabled for now
    # layout = definition_values.get('layout', None)
    # if layout is None:
    #   error_message = 'missing layout'
    #   raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object.metadata = definition_values.get('metadata', {})

    attributes = definition_values.get('attributes', None)
    if attributes:
      unsupported_attributes = set(attributes.keys()).difference(
          self._SUPPORTED_ATTRIBUTES_FORMAT)
      if unsupported_attributes:
        error_message = 'unsupported attributes: {0:s}'.format(
            ', '.join(unsupported_attributes))
        raise errors.DefinitionReaderError(definition_name, error_message)

      byte_order = attributes.get('byte_order', definitions.BYTE_ORDER_NATIVE)
      if byte_order not in definitions.BYTE_ORDERS:
        error_message = 'unsupported byte-order attribute: {0!s}'.format(
            byte_order)
        raise errors.DefinitionReaderError(definition_name, error_message)

      definition_object.byte_order = byte_order

    return definition_object
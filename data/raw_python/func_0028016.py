def _ReadStorageDataTypeDefinition(
      self, definitions_registry, definition_values, data_type_definition_class,
      definition_name, supported_attributes, is_member=False):
    """Reads a storage data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      data_type_definition_class (str): data type definition class.
      definition_name (str): name of the definition.
      supported_attributes (set[str]): names of the supported attributes.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      StorageDataTypeDefinition: storage data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if is_member:
      supported_definition_values = (
          self._SUPPORTED_DEFINITION_VALUES_MEMBER_DATA_TYPE)
    else:
      supported_definition_values = (
          self._SUPPORTED_DEFINITION_VALUES_STORAGE_DATA_TYPE)

    definition_object = self._ReadDataTypeDefinition(
        definitions_registry, definition_values, data_type_definition_class,
        definition_name, supported_definition_values)

    attributes = definition_values.get('attributes', None)
    if attributes:
      unsupported_attributes = set(attributes.keys()).difference(
          supported_attributes)
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
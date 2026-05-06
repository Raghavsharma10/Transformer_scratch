def _ReadFixedSizeDataTypeDefinition(
      self, definitions_registry, definition_values, data_type_definition_class,
      definition_name, supported_attributes,
      default_size=definitions.SIZE_NATIVE, default_units='bytes',
      is_member=False, supported_size_values=None):
    """Reads a fixed-size data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      data_type_definition_class (str): data type definition class.
      definition_name (str): name of the definition.
      supported_attributes (set[str]): names of the supported attributes.
      default_size (Optional[int]): default size.
      default_units (Optional[str]): default units.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.
      supported_size_values (Optional[tuple[int]]): supported size values,
          or None if not set.

    Returns:
      FixedSizeDataTypeDefinition: fixed-size data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    definition_object = self._ReadStorageDataTypeDefinition(
        definitions_registry, definition_values, data_type_definition_class,
        definition_name, supported_attributes, is_member=is_member)

    attributes = definition_values.get('attributes', None)
    if attributes:
      size = attributes.get('size', default_size)
      if size != definitions.SIZE_NATIVE:
        try:
          int(size)
        except ValueError:
          error_message = 'unuspported size attribute: {0!s}'.format(size)
          raise errors.DefinitionReaderError(definition_name, error_message)

        if supported_size_values and size not in supported_size_values:
          error_message = 'unuspported size value: {0!s}'.format(size)
          raise errors.DefinitionReaderError(definition_name, error_message)

      definition_object.size = size
      definition_object.units = attributes.get('units', default_units)

    return definition_object
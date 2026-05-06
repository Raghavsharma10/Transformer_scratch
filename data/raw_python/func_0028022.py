def _ReadUUIDDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads an UUID data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      UUIDDataTypeDefinition: UUID data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    return self._ReadFixedSizeDataTypeDefinition(
        definitions_registry, definition_values,
        data_types.UUIDDefinition, definition_name,
        self._SUPPORTED_ATTRIBUTES_FIXED_SIZE_DATA_TYPE, default_size=16,
        is_member=is_member, supported_size_values=(16, ))
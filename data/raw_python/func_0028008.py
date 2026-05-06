def _ReadFloatingPointDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a floating-point data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      FloatingPointDefinition: floating-point data type definition.
    """
    return self._ReadFixedSizeDataTypeDefinition(
        definitions_registry, definition_values,
        data_types.FloatingPointDefinition, definition_name,
        self._SUPPORTED_ATTRIBUTES_FIXED_SIZE_DATA_TYPE,
        is_member=is_member, supported_size_values=(4, 8))
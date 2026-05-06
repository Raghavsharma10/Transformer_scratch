def _ReadBooleanDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a boolean data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      BooleanDataTypeDefinition: boolean data type definition.
    """
    return self._ReadFixedSizeDataTypeDefinition(
        definitions_registry, definition_values,
        data_types.BooleanDefinition, definition_name,
        self._SUPPORTED_ATTRIBUTES_BOOLEAN, is_member=is_member,
        supported_size_values=(1, 2, 4))
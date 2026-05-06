def _ReadStructureDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a structure data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      StructureDefinition: structure data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if is_member:
      error_message = 'data type not supported as member'
      raise errors.DefinitionReaderError(definition_name, error_message)

    return self._ReadDataTypeDefinitionWithMembers(
        definitions_registry, definition_values, data_types.StructureDefinition,
        definition_name, supports_conditions=True)
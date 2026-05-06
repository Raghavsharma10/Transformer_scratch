def _ReadStructureFamilyDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads a structure family data type definition.

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

    definition_object = self._ReadLayoutDataTypeDefinition(
        definitions_registry, definition_values,
        data_types.StructureFamilyDefinition, definition_name,
        self._SUPPORTED_DEFINITION_VALUES_STRUCTURE_FAMILY)

    runtime = definition_values.get('runtime', None)
    if not runtime:
      error_message = 'missing runtime'
      raise errors.DefinitionReaderError(definition_name, error_message)

    runtime_data_type_definition = definitions_registry.GetDefinitionByName(
        runtime)
    if not runtime_data_type_definition:
      error_message = 'undefined runtime: {0:s}.'.format(runtime)
      raise errors.DefinitionReaderError(definition_name, error_message)

    if runtime_data_type_definition.family_definition:
      error_message = 'runtime: {0:s} already part of a family.'.format(runtime)
      raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object.AddRuntimeDefinition(runtime_data_type_definition)

    members = definition_values.get('members', None)
    if not members:
      error_message = 'missing members'
      raise errors.DefinitionReaderError(definition_name, error_message)

    for member in members:
      member_data_type_definition = definitions_registry.GetDefinitionByName(
          member)
      if not member_data_type_definition:
        error_message = 'undefined member: {0:s}.'.format(member)
        raise errors.DefinitionReaderError(definition_name, error_message)

      if member_data_type_definition.family_definition:
        error_message = 'member: {0:s} already part of a family.'.format(member)
        raise errors.DefinitionReaderError(definition_name, error_message)

      definition_object.AddMemberDefinition(member_data_type_definition)

    return definition_object
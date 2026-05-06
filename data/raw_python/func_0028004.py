def _ReadDataTypeDefinitionWithMembers(
      self, definitions_registry, definition_values,
      data_type_definition_class, definition_name, supports_conditions=False):
    """Reads a data type definition with members.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      data_type_definition_class (str): data type definition class.
      definition_name (str): name of the definition.
      supports_conditions (Optional[bool]): True if conditions are supported
          by the data type definition.

    Returns:
      StringDefinition: string data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    members = definition_values.get('members', None)
    if not members:
      error_message = 'missing members'
      raise errors.DefinitionReaderError(definition_name, error_message)

    supported_definition_values = (
        self._SUPPORTED_DEFINITION_VALUES_STORAGE_DATA_TYPE_WITH_MEMBERS)
    definition_object = self._ReadDataTypeDefinition(
        definitions_registry, definition_values, data_type_definition_class,
        definition_name, supported_definition_values)

    attributes = definition_values.get('attributes', None)
    if attributes:
      unsupported_attributes = set(attributes.keys()).difference(
          self._SUPPORTED_ATTRIBUTES_STORAGE_DATA_TYPE)
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

    for member in members:
      section = member.get('section', None)
      if section:
        member_section_definition = data_types.MemberSectionDefinition(section)
        definition_object.AddSectionDefinition(member_section_definition)
      else:
        member_data_type_definition = self._ReadMemberDataTypeDefinitionMember(
            definitions_registry, member, definition_object.name,
            supports_conditions=supports_conditions)
        definition_object.AddMemberDefinition(member_data_type_definition)

    return definition_object
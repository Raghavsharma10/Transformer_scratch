def _ReadEnumerationDataTypeDefinition(
      self, definitions_registry, definition_values, definition_name,
      is_member=False):
    """Reads an enumeration data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      is_member (Optional[bool]): True if the data type definition is a member
          data type definition.

    Returns:
      EnumerationDataTypeDefinition: enumeration data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if is_member:
      error_message = 'data type not supported as member'
      raise errors.DefinitionReaderError(definition_name, error_message)

    values = definition_values.get('values')
    if not values:
      error_message = 'missing values'
      raise errors.DefinitionReaderError(definition_name, error_message)

    definition_object = self._ReadSemanticDataTypeDefinition(
        definitions_registry, definition_values,
        data_types.EnumerationDefinition, definition_name,
        self._SUPPORTED_DEFINITION_VALUES_ENUMERATION)

    last_name = None
    for enumeration_value in values:
      aliases = enumeration_value.get('aliases', None)
      description = enumeration_value.get('description', None)
      name = enumeration_value.get('name', None)
      number = enumeration_value.get('number', None)

      if not name or number is None:
        if last_name:
          error_location = 'after: {0:s}'.format(last_name)
        else:
          error_location = 'at start'

        error_message = '{0:s} missing name or number'.format(error_location)
        raise errors.DefinitionReaderError(definition_name, error_message)

      else:
        try:
          definition_object.AddValue(
              name, number, aliases=aliases, description=description)
        except KeyError as exception:
          error_message = '{0!s}'.format(exception)
          raise errors.DefinitionReaderError(definition_name, error_message)

      last_name = name

    return definition_object
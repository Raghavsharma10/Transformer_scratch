def _ReadMemberDataTypeDefinitionMember(
      self, definitions_registry, definition_values, definition_name,
      supports_conditions=False):
    """Reads a member data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      definition_name (str): name of the definition.
      supports_conditions (Optional[bool]): True if conditions are supported
          by the data type definition.

    Returns:
      DataTypeDefinition: structure member data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    if not definition_values:
      error_message = 'invalid structure member missing definition values'
      raise errors.DefinitionReaderError(definition_name, error_message)

    name = definition_values.get('name', None)
    type_indicator = definition_values.get('type', None)

    if not name and type_indicator != definitions.TYPE_INDICATOR_UNION:
      error_message = 'invalid structure member missing name'
      raise errors.DefinitionReaderError(definition_name, error_message)

    # TODO: detect duplicate names.

    data_type = definition_values.get('data_type', None)

    type_values = (data_type, type_indicator)
    type_values = [value for value in type_values if value is not None]

    if not type_values:
      error_message = (
          'invalid structure member: {0:s} both data type and type are '
          'missing').format(name or '<NAMELESS>')
      raise errors.DefinitionReaderError(definition_name, error_message)

    if len(type_values) > 1:
      error_message = (
          'invalid structure member: {0:s} data type and type not allowed to '
          'be set at the same time').format(name or '<NAMELESS>')
      raise errors.DefinitionReaderError(definition_name, error_message)

    condition = definition_values.get('condition', None)
    if not supports_conditions and condition:
      error_message = (
          'invalid structure member: {0:s} unsupported condition').format(
              name or '<NAMELESS>')
      raise errors.DefinitionReaderError(definition_name, error_message)

    value = definition_values.get('value', None)
    values = definition_values.get('values', None)

    if None not in (value, values):
      error_message = (
          'invalid structure member: {0:s} value and values not allowed to '
          'be set at the same time').format(name or '<NAMELESS>')
      raise errors.DefinitionReaderError(definition_name, error_message)

    if value:
      values = [value]

    supported_values = None
    if values:
      supported_values = []
      for value in values:
        if isinstance(value, py2to3.UNICODE_TYPE):
          value = value.encode('ascii')

        supported_values.append(value)

    if type_indicator is not None:
      data_type_callback = self._DATA_TYPE_CALLBACKS.get(type_indicator, None)
      if data_type_callback:
        data_type_callback = getattr(self, data_type_callback, None)
      if not data_type_callback:
        error_message = 'unuspported data type definition: {0:s}.'.format(
            type_indicator)
        raise errors.DefinitionReaderError(name, error_message)

      try:
        data_type_definition = data_type_callback(
            definitions_registry, definition_values, name, is_member=True)
      except errors.DefinitionReaderError as exception:
        error_message = 'in: {0:s} {1:s}'.format(
            exception.name or '<NAMELESS>', exception.message)
        raise errors.DefinitionReaderError(definition_name, error_message)

      if condition or supported_values:
        definition_object = data_types.MemberDataTypeDefinition(
            name, data_type_definition, condition=condition,
            values=supported_values)
      else:
        definition_object = data_type_definition

    elif data_type is not None:
      data_type_definition = definitions_registry.GetDefinitionByName(
          data_type)
      if not data_type_definition:
        error_message = (
            'invalid structure member: {0:s} undefined data type: '
            '{1:s}').format(name or '<NAMELESS>', data_type)
        raise errors.DefinitionReaderError(definition_name, error_message)

      unsupported_definition_values = set(definition_values.keys()).difference(
          self._SUPPORTED_DEFINITION_VALUES_MEMBER_DATA_TYPE)
      if unsupported_definition_values:
        error_message = 'unsupported definition values: {0:s}'.format(
            ', '.join(unsupported_definition_values))
        raise errors.DefinitionReaderError(definition_name, error_message)

      aliases = definition_values.get('aliases', None)
      description = definition_values.get('description', None)

      definition_object = data_types.MemberDataTypeDefinition(
          name, data_type_definition, aliases=aliases, condition=condition,
          data_type=data_type, description=description, values=supported_values)

    return definition_object
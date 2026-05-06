def _ReadElementSequenceDataTypeDefinition(
      self, definitions_registry, definition_values,
      data_type_definition_class, definition_name, supported_definition_values):
    """Reads an element sequence data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      data_type_definition_class (str): data type definition class.
      definition_name (str): name of the definition.
      supported_definition_values (set[str]): names of the supported definition
          values.

    Returns:
      SequenceDefinition: sequence data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    unsupported_definition_values = set(definition_values.keys()).difference(
        supported_definition_values)
    if unsupported_definition_values:
      error_message = 'unsupported definition values: {0:s}'.format(
          ', '.join(unsupported_definition_values))
      raise errors.DefinitionReaderError(definition_name, error_message)

    element_data_type = definition_values.get('element_data_type', None)
    if not element_data_type:
      error_message = 'missing element data type'
      raise errors.DefinitionReaderError(definition_name, error_message)

    elements_data_size = definition_values.get('elements_data_size', None)
    elements_terminator = definition_values.get('elements_terminator', None)
    number_of_elements = definition_values.get('number_of_elements', None)

    size_values = (elements_data_size, elements_terminator, number_of_elements)
    size_values = [value for value in size_values if value is not None]

    if not size_values:
      error_message = (
          'missing element data size, elements terminator and number of '
          'elements')
      raise errors.DefinitionReaderError(definition_name, error_message)

    if len(size_values) > 1:
      error_message = (
          'element data size, elements terminator and number of elements '
          'not allowed to be set at the same time')
      raise errors.DefinitionReaderError(definition_name, error_message)

    element_data_type_definition = definitions_registry.GetDefinitionByName(
        element_data_type)
    if not element_data_type_definition:
      error_message = 'undefined element data type: {0:s}.'.format(
          element_data_type)
      raise errors.DefinitionReaderError(definition_name, error_message)

    element_byte_size = element_data_type_definition.GetByteSize()
    element_type_indicator = element_data_type_definition.TYPE_INDICATOR
    if not element_byte_size and element_type_indicator != (
        definitions.TYPE_INDICATOR_STRING):
      error_message = (
          'unsupported variable size element data type: {0:s}'.format(
              element_data_type))
      raise errors.DefinitionReaderError(definition_name, error_message)

    aliases = definition_values.get('aliases', None)
    description = definition_values.get('description', None)
    urls = definition_values.get('urls', None)

    definition_object = data_type_definition_class(
        definition_name, element_data_type_definition, aliases=aliases,
        data_type=element_data_type, description=description, urls=urls)

    if elements_data_size is not None:
      try:
        definition_object.elements_data_size = int(elements_data_size)
      except ValueError:
        definition_object.elements_data_size_expression = elements_data_size

    elif elements_terminator is not None:
      if isinstance(elements_terminator, py2to3.UNICODE_TYPE):
        elements_terminator = elements_terminator.encode('ascii')

      definition_object.elements_terminator = elements_terminator

    elif number_of_elements is not None:
      try:
        definition_object.number_of_elements = int(number_of_elements)
      except ValueError:
        definition_object.number_of_elements_expression = number_of_elements

    return definition_object
def _GetElementDataTypeDefinition(self, data_type_definition):
    """Retrieves the element data type definition.

    Args:
      data_type_definition (DataTypeDefinition): data type definition.

    Returns:
      DataTypeDefinition: element data type definition.

    Raises:
      FormatError: if the element data type cannot be determined from the data
          type definition.
    """
    if not data_type_definition:
      raise errors.FormatError('Missing data type definition')

    element_data_type_definition = getattr(
        data_type_definition, 'element_data_type_definition', None)
    if not element_data_type_definition:
      raise errors.FormatError(
          'Invalid data type definition missing element')

    return element_data_type_definition
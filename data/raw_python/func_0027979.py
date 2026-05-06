def _GetAttributeNames(self, data_type_definition):
    """Determines the attribute (or field) names of the members.

    Args:
      data_type_definition (DataTypeDefinition): data type definition.

    Returns:
      list[str]: attribute names.

    Raises:
      FormatError: if the attribute names cannot be determined from the data
          type definition.
    """
    if not data_type_definition:
      raise errors.FormatError('Missing data type definition')

    attribute_names = []
    for member_definition in data_type_definition.members:
      attribute_names.append(member_definition.name)

    return attribute_names
def _ValidateDataTypeDefinition(cls, data_type_definition):
    """Validates the data type definition.

    Args:
      data_type_definition (DataTypeDefinition): data type definition.

    Raises:
      ValueError: if the data type definition is not considered valid.
    """
    if not cls._IsIdentifier(data_type_definition.name):
      raise ValueError(
          'Data type definition name: {0!s} not a valid identifier'.format(
              data_type_definition.name))

    if keyword.iskeyword(data_type_definition.name):
      raise ValueError(
          'Data type definition name: {0!s} matches keyword'.format(
              data_type_definition.name))

    members = getattr(data_type_definition, 'members', None)
    if not members:
      raise ValueError(
          'Data type definition name: {0!s} missing members'.format(
              data_type_definition.name))

    defined_attribute_names = set()

    for member_definition in members:
      attribute_name = member_definition.name

      if not cls._IsIdentifier(attribute_name):
        raise ValueError('Attribute name: {0!s} not a valid identifier'.format(
            attribute_name))

      if attribute_name.startswith('_'):
        raise ValueError('Attribute name: {0!s} starts with underscore'.format(
            attribute_name))

      if keyword.iskeyword(attribute_name):
        raise ValueError('Attribute name: {0!s} matches keyword'.format(
            attribute_name))

      if attribute_name in defined_attribute_names:
        raise ValueError('Attribute name: {0!s} already defined'.format(
            attribute_name))

      defined_attribute_names.add(attribute_name)
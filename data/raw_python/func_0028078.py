def _CreateClassTemplate(cls, data_type_definition):
    """Creates the class template.

    Args:
      data_type_definition (DataTypeDefinition): data type definition.

    Returns:
      str: class template.
    """
    type_name = data_type_definition.name

    type_description = data_type_definition.description or type_name
    while type_description.endswith('.'):
      type_description = type_description[:-1]

    class_attributes_description = []
    init_arguments = []
    instance_attributes = []

    for member_definition in data_type_definition.members:
      attribute_name = member_definition.name

      description = member_definition.description or attribute_name
      while description.endswith('.'):
        description = description[:-1]

      member_data_type = getattr(member_definition, 'member_data_type', '')
      if isinstance(member_definition, data_types.MemberDataTypeDefinition):
        member_definition = member_definition.member_data_type_definition

      member_type_indicator = member_definition.TYPE_INDICATOR
      if member_type_indicator == definitions.TYPE_INDICATOR_SEQUENCE:
        element_type_indicator = member_definition.element_data_type
        member_type_indicator = 'tuple[{0:s}]'.format(element_type_indicator)
      else:
        member_type_indicator = cls._PYTHON_NATIVE_TYPES.get(
            member_type_indicator, member_data_type)

      argument = '{0:s}=None'.format(attribute_name)

      definition = '    self.{0:s} = {0:s}'.format(attribute_name)

      description = '    {0:s} ({1:s}): {2:s}.'.format(
          attribute_name, member_type_indicator, description)

      class_attributes_description.append(description)
      init_arguments.append(argument)
      instance_attributes.append(definition)

    class_attributes_description = '\n'.join(
        sorted(class_attributes_description))
    init_arguments = ', '.join(init_arguments)
    instance_attributes = '\n'.join(sorted(instance_attributes))

    template_values = {
        'class_attributes_description': class_attributes_description,
        'init_arguments': init_arguments,
        'instance_attributes': instance_attributes,
        'type_description': type_description,
        'type_name': type_name}

    return cls._CLASS_TEMPLATE.format(**template_values)
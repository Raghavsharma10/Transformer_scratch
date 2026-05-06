def _GetFormatErrorLocation(
      self, yaml_definition, last_definition_object):
    """Retrieves a format error location.

    Args:
      yaml_definition (dict[str, object]): current YAML definition.
      last_definition_object (DataTypeDefinition): previous data type
          definition.

    Returns:
      str: format error location.
    """
    name = yaml_definition.get('name', None)
    if name:
      error_location = 'in: {0:s}'.format(name or '<NAMELESS>')
    elif last_definition_object:
      error_location = 'after: {0:s}'.format(last_definition_object.name)
    else:
      error_location = 'at start'

    return error_location
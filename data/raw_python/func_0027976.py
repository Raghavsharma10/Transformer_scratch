def _CheckCompositeMap(self, data_type_definition):
    """Determines if the data type definition needs a composite map.

    Args:
      data_type_definition (DataTypeDefinition): structure data type definition.

    Returns:
      bool: True if a composite map is needed, False otherwise.

    Raises:
      FormatError: if a composite map is needed cannot be determined from the
          data type definition.
    """
    if not data_type_definition:
      raise errors.FormatError('Missing data type definition')

    members = getattr(data_type_definition, 'members', None)
    if not members:
      raise errors.FormatError('Invalid data type definition missing members')

    is_composite_map = False
    last_member_byte_order = data_type_definition.byte_order

    for member_definition in members:
      if member_definition.IsComposite():
        is_composite_map = True
        break

      # TODO: check for padding type
      # TODO: determine if padding type can be defined as linear
      if (last_member_byte_order != definitions.BYTE_ORDER_NATIVE and
          member_definition.byte_order != definitions.BYTE_ORDER_NATIVE and
          last_member_byte_order != member_definition.byte_order):
        is_composite_map = True
        break

      last_member_byte_order = member_definition.byte_order

    return is_composite_map
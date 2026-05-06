def _GetMemberDataTypeMaps(self, data_type_definition, data_type_map_cache):
    """Retrieves the member data type maps.

    Args:
      data_type_definition (DataTypeDefinition): data type definition.
      data_type_map_cache (dict[str, DataTypeMap]): cached data type maps.

    Returns:
      list[DataTypeMap]: member data type maps.

    Raises:
      FormatError: if the data type maps cannot be determined from the data
          type definition.
    """
    if not data_type_definition:
      raise errors.FormatError('Missing data type definition')

    members = getattr(data_type_definition, 'members', None)
    if not members:
      raise errors.FormatError('Invalid data type definition missing members')

    data_type_maps = []

    members_data_size = 0
    for member_definition in members:
      if isinstance(member_definition, data_types.MemberDataTypeDefinition):
        member_definition = member_definition.member_data_type_definition

      if (data_type_definition.byte_order != definitions.BYTE_ORDER_NATIVE and
          member_definition.byte_order == definitions.BYTE_ORDER_NATIVE):
        # Make a copy of the data type definition where byte-order can be
        # safely changed.
        member_definition = copy.copy(member_definition)
        member_definition.name = '_{0:s}_{1:s}'.format(
            data_type_definition.name, member_definition.name)
        member_definition.byte_order = data_type_definition.byte_order

      if member_definition.name not in data_type_map_cache:
        data_type_map = DataTypeMapFactory.CreateDataTypeMapByType(
            member_definition)
        data_type_map_cache[member_definition.name] = data_type_map

      data_type_map = data_type_map_cache[member_definition.name]
      if members_data_size is not None:
        if not isinstance(member_definition, data_types.PaddingDefinition):
          byte_size = member_definition.GetByteSize()
        else:
          _, byte_size = divmod(
              members_data_size, member_definition.alignment_size)
          if byte_size > 0:
            byte_size = member_definition.alignment_size - byte_size

          data_type_map.byte_size = byte_size

        if byte_size is None:
          members_data_size = None
        else:
          members_data_size += byte_size

      data_type_maps.append(data_type_map)

    return data_type_maps
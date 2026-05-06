def _LinearMapByteStream(
      self, byte_stream, byte_offset=0, context=None, **unused_kwargs):
    """Maps a data type sequence on a byte stream.

    Args:
      byte_stream (bytes): byte stream.
      byte_offset (Optional[int]): offset into the byte stream where to start.
      context (Optional[DataTypeMapContext]): data type map context.

    Returns:
      object: mapped value.

    Raises:
      MappingError: if the data type definition cannot be mapped on
          the byte stream.
    """
    members_data_size = self._data_type_definition.GetByteSize()
    self._CheckByteStreamSize(byte_stream, byte_offset, members_data_size)

    try:
      struct_tuple = self._operation.ReadFrom(byte_stream[byte_offset:])
      struct_values = []
      for attribute_index, value in enumerate(struct_tuple):
        data_type_map = self._data_type_maps[attribute_index]
        member_definition = self._data_type_definition.members[attribute_index]

        value = data_type_map.MapValue(value)

        supported_values = getattr(member_definition, 'values', None)
        if supported_values and value not in supported_values:
          raise errors.MappingError(
              'Value: {0!s} not in supported values: {1:s}'.format(
                  value, ', '.join([
                      '{0!s}'.format(value) for value in supported_values])))

        struct_values.append(value)

      mapped_value = self._structure_values_class(*struct_values)

    except Exception as exception:
      error_string = (
          'Unable to read: {0:s} from byte stream at offset: {1:d} '
          'with error: {2!s}').format(
              self._data_type_definition.name, byte_offset, exception)
      raise errors.MappingError(error_string)

    if context:
      context.byte_size = members_data_size

    return mapped_value
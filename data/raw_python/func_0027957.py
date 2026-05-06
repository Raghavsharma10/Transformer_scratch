def MapByteStream(
      self, byte_stream, byte_offset=0, context=None, **unused_kwargs):
    """Maps the data type on a byte stream.

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
    data_type_size = self._data_type_definition.GetByteSize()
    self._CheckByteStreamSize(byte_stream, byte_offset, data_type_size)

    try:
      struct_tuple = self._operation.ReadFrom(byte_stream[byte_offset:])
      mapped_value = self.MapValue(*struct_tuple)

    except Exception as exception:
      error_string = (
          'Unable to read: {0:s} from byte stream at offset: {1:d} '
          'with error: {2!s}').format(
              self._data_type_definition.name, byte_offset, exception)
      raise errors.MappingError(error_string)

    if context:
      context.byte_size = data_type_size

    return mapped_value
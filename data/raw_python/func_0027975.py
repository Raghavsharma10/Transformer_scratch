def MapByteStream(self, byte_stream, byte_offset=0, **kwargs):
    """Maps the data type on a byte stream.

    Args:
      byte_stream (bytes): byte stream.
      byte_offset (Optional[int]): offset into the byte stream where to start.

    Returns:
      str: mapped values.

    Raises:
      MappingError: if the data type definition cannot be mapped on
          the byte stream.
    """
    byte_stream = super(StringMap, self).MapByteStream(
        byte_stream, byte_offset=byte_offset, **kwargs)

    if self._HasElementsTerminator():
      # Remove the elements terminator and any trailing data from
      # the byte stream.
      elements_terminator = self._data_type_definition.elements_terminator
      elements_terminator_size = len(elements_terminator)

      byte_offset = 0
      byte_stream_size = len(byte_stream)

      while byte_offset < byte_stream_size:
        end_offset = byte_offset + elements_terminator_size
        if byte_stream[byte_offset:end_offset] == elements_terminator:
          break

        byte_offset += elements_terminator_size

      byte_stream = byte_stream[:byte_offset]

    try:
      return byte_stream.decode(self._data_type_definition.encoding)

    except Exception as exception:
      error_string = (
          'Unable to read: {0:s} from byte stream at offset: {1:d} '
          'with error: {2!s}').format(
              self._data_type_definition.name, byte_offset, exception)
      raise errors.MappingError(error_string)
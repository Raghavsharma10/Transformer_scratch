def _CheckByteStreamSize(self, byte_stream, byte_offset, data_type_size):
    """Checks if the byte stream is large enough for the data type.

    Args:
      byte_stream (bytes): byte stream.
      byte_offset (int): offset into the byte stream where to start.
      data_type_size (int): data type size.

    Raises:
      ByteStreamTooSmallError: if the byte stream is too small.
      MappingError: if the size of the byte stream cannot be determined.
    """
    try:
      byte_stream_size = len(byte_stream)

    except Exception as exception:
      raise errors.MappingError(exception)

    if byte_stream_size - byte_offset < data_type_size:
      raise errors.ByteStreamTooSmallError(
          'Byte stream too small requested: {0:d} available: {1:d}'.format(
              data_type_size, byte_stream_size))
def FoldByteStream(self, mapped_value, **kwargs):
    """Folds the data type into a byte stream.

    Args:
      mapped_value (object): mapped value.

    Returns:
      bytes: byte stream.

    Raises:
      FoldingError: if the data type definition cannot be folded into
          the byte stream.
    """
    try:
      byte_stream = mapped_value.encode(self._data_type_definition.encoding)

    except Exception as exception:
      error_string = (
          'Unable to write: {0:s} to byte stream with error: {1!s}').format(
              self._data_type_definition.name, exception)
      raise errors.MappingError(error_string)

    return super(StringMap, self).FoldByteStream(byte_stream, **kwargs)
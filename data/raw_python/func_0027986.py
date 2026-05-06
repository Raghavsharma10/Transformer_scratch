def MapByteStream(self, byte_stream, **unused_kwargs):  # pylint: disable=redundant-returns-doc
    """Maps the data type on a byte stream.

    Args:
      byte_stream (bytes): byte stream.

    Returns:
      object: mapped value.

    Raises:
      MappingError: if the data type definition cannot be mapped on
          the byte stream.
    """
    raise errors.MappingError(
        'Unable to map {0:s} data type to byte stream'.format(
            self._data_type_definition.TYPE_INDICATOR))
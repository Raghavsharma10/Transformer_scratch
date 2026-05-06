def FoldByteStream(self, mapped_value, **unused_kwargs):
    """Folds the data type into a byte stream.

    Args:
      mapped_value (object): mapped value.

    Returns:
      bytes: byte stream.

    Raises:
      FoldingError: if the data type definition cannot be folded into
          the byte stream.
    """
    value = None

    try:
      if self._byte_order == definitions.BYTE_ORDER_BIG_ENDIAN:
        value = mapped_value.bytes
      elif self._byte_order == definitions.BYTE_ORDER_LITTLE_ENDIAN:
        value = mapped_value.bytes_le

    except Exception as exception:
      error_string = (
          'Unable to write: {0:s} to byte stream with error: {1!s}').format(
              self._data_type_definition.name, exception)
      raise errors.FoldingError(error_string)

    return value
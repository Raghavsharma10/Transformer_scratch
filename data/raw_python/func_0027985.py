def FoldByteStream(self, mapped_value, **unused_kwargs):  # pylint: disable=redundant-returns-doc
    """Folds the data type into a byte stream.

    Args:
      mapped_value (object): mapped value.

    Returns:
      bytes: byte stream.

    Raises:
      FoldingError: if the data type definition cannot be folded into
          the byte stream.
    """
    raise errors.FoldingError(
        'Unable to fold {0:s} data type into byte stream'.format(
            self._data_type_definition.TYPE_INDICATOR))
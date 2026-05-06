def FoldByteStream(self, mapped_value, context=None, **unused_kwargs):
    """Folds the data type into a byte stream.

    Args:
      mapped_value (object): mapped value.
      context (Optional[DataTypeMapContext]): data type map context.

    Returns:
      bytes: byte stream.

    Raises:
      FoldingError: if the data type definition cannot be folded into
          the byte stream.
    """
    elements_data_size = self._CalculateElementsDataSize(context)
    if elements_data_size is not None:
      if elements_data_size != len(mapped_value):
        raise errors.FoldingError(
            'Mismatch between elements data size and mapped value size')

    elif not self._HasElementsTerminator():
      raise errors.FoldingError('Unable to determine elements data size')

    else:
      elements_terminator = self._data_type_definition.elements_terminator
      elements_terminator_size = len(elements_terminator)
      if mapped_value[-elements_terminator_size:] != elements_terminator:
        mapped_value = b''.join([mapped_value, elements_terminator])

    return mapped_value
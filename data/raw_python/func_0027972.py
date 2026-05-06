def MapByteStream(
      self, byte_stream, byte_offset=0, context=None, **unused_kwargs):
    """Maps the data type on a byte stream.

    Args:
      byte_stream (bytes): byte stream.
      byte_offset (Optional[int]): offset into the byte stream where to start.
      context (Optional[DataTypeMapContext]): data type map context.

    Returns:
      tuple[object, ...]: mapped values.

    Raises:
      MappingError: if the data type definition cannot be mapped on
          the byte stream.
    """
    context_state = getattr(context, 'state', {})

    size_hints = context_state.get('size_hints', {})

    elements_data_size = self._CalculateElementsDataSize(context)
    if elements_data_size is not None:
      self._CheckByteStreamSize(byte_stream, byte_offset, elements_data_size)

    elif not self._HasElementsTerminator():
      raise errors.MappingError(
          'Unable to determine elements data size and missing elements '
          'terminator')

    else:
      byte_stream_size = len(byte_stream)

      element_byte_size = self._element_data_type_definition.GetByteSize()
      elements_data_offset = byte_offset
      next_elements_data_offset = elements_data_offset + element_byte_size

      elements_terminator = self._data_type_definition.elements_terminator
      element_value = byte_stream[
          elements_data_offset:next_elements_data_offset]

      while byte_stream[elements_data_offset:]:
        elements_data_offset = next_elements_data_offset
        if element_value == elements_terminator:
          elements_data_size = elements_data_offset - byte_offset
          break

        next_elements_data_offset += element_byte_size
        element_value = byte_stream[
            elements_data_offset:next_elements_data_offset]

      if element_value != elements_terminator:
        size_hints[self._data_type_definition.name] = DataTypeMapSizeHint(
            byte_stream_size - byte_offset)

        context_state['size_hints'] = size_hints

        error_string = (
            'Unable to read: {0:s} from byte stream at offset: {1:d} '
            'with error: unable to find elements terminator').format(
                self._data_type_definition.name, byte_offset)
        raise errors.ByteStreamTooSmallError(error_string)

    if context:
      context.byte_size = elements_data_size

      size_hints[self._data_type_definition.name] = DataTypeMapSizeHint(
          elements_data_size, is_complete=True)

      context_state['size_hints'] = size_hints

    return byte_stream[byte_offset:byte_offset + elements_data_size]
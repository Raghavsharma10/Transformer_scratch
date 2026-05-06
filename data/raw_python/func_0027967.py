def _CompositeMapByteStream(
      self, byte_stream, byte_offset=0, context=None, **unused_kwargs):
    """Maps a sequence of composite data types on a byte stream.

    Args:
      byte_stream (bytes): byte stream.
      byte_offset (Optional[int]): offset into the byte stream where to start.
      context (Optional[DataTypeMapContext]): data type map context.

    Returns:
      tuple[object, ...]: mapped values.

    Raises:
      ByteStreamTooSmallError: if the byte stream is too small.
      MappingError: if the data type definition cannot be mapped on
          the byte stream.
    """
    elements_data_size = None
    elements_terminator = None
    number_of_elements = None

    if self._HasElementsDataSize():
      elements_data_size = self._EvaluateElementsDataSize(context)

      element_byte_size = self._element_data_type_definition.GetByteSize()
      if element_byte_size is not None:
        number_of_elements, _ = divmod(elements_data_size, element_byte_size)
      else:
        elements_terminator = (
            self._element_data_type_definition.elements_terminator)

    elif self._HasElementsTerminator():
      elements_terminator = self._data_type_definition.elements_terminator

    elif self._HasNumberOfElements():
      number_of_elements = self._EvaluateNumberOfElements(context)

    if elements_terminator is None and number_of_elements is None:
      raise errors.MappingError(
          'Unable to determine element terminator or number of elements')

    context_state = getattr(context, 'state', {})

    elements_data_offset = context_state.get('elements_data_offset', 0)
    element_index = context_state.get('element_index', 0)
    element_value = None
    mapped_values = context_state.get('mapped_values', [])
    size_hints = context_state.get('size_hints', {})
    subcontext = context_state.get('context', None)

    if not subcontext:
      subcontext = DataTypeMapContext()

    try:
      while byte_stream[byte_offset:]:
        if (number_of_elements is not None and
            element_index == number_of_elements):
          break

        if (elements_data_size is not None and
            elements_data_offset >= elements_data_size):
          break

        element_value = self._element_data_type_map.MapByteStream(
            byte_stream, byte_offset=byte_offset, context=subcontext)

        byte_offset += subcontext.byte_size
        elements_data_offset += subcontext.byte_size
        element_index += 1
        mapped_values.append(element_value)

        if (elements_terminator is not None and
            element_value == elements_terminator):
          break

    except errors.ByteStreamTooSmallError as exception:
      context_state['context'] = subcontext
      context_state['elements_data_offset'] = elements_data_offset
      context_state['element_index'] = element_index
      context_state['mapped_values'] = mapped_values
      raise errors.ByteStreamTooSmallError(exception)

    except Exception as exception:
      raise errors.MappingError(exception)

    if number_of_elements is not None and element_index != number_of_elements:
      context_state['context'] = subcontext
      context_state['elements_data_offset'] = elements_data_offset
      context_state['element_index'] = element_index
      context_state['mapped_values'] = mapped_values

      error_string = (
          'Unable to read: {0:s} from byte stream at offset: {1:d} '
          'with error: missing element: {2:d}').format(
              self._data_type_definition.name, byte_offset, element_index - 1)
      raise errors.ByteStreamTooSmallError(error_string)

    if (elements_terminator is not None and
        element_value != elements_terminator and (
            elements_data_size is None or
            elements_data_offset < elements_data_size)):
      byte_stream_size = len(byte_stream)

      size_hints[self._data_type_definition.name] = DataTypeMapSizeHint(
          byte_stream_size - byte_offset)

      context_state['context'] = subcontext
      context_state['elements_data_offset'] = elements_data_offset
      context_state['element_index'] = element_index
      context_state['mapped_values'] = mapped_values
      context_state['size_hints'] = size_hints

      error_string = (
          'Unable to read: {0:s} from byte stream at offset: {1:d} '
          'with error: unable to find elements terminator').format(
              self._data_type_definition.name, byte_offset)
      raise errors.ByteStreamTooSmallError(error_string)

    if context:
      context.byte_size = elements_data_offset
      context.state = {}

    return tuple(mapped_values)
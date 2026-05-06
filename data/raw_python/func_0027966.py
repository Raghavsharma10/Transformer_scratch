def GetSizeHint(self, context=None, **unused_kwargs):
    """Retrieves a hint about the size.

    Args:
      context (Optional[DataTypeMapContext]): data type map context, used to
          determine the size hint.

    Returns:
      int: hint of the number of bytes needed from the byte stream or None.
    """
    context_state = getattr(context, 'state', {})

    elements_data_size = self.GetByteSize()
    if elements_data_size:
      return elements_data_size

    try:
      elements_data_size = self._CalculateElementsDataSize(context)
    except errors.MappingError:
      pass

    if elements_data_size is None and self._HasElementsTerminator():
      size_hints = context_state.get('size_hints', {})
      size_hint = size_hints.get(self._data_type_definition.name, None)

      elements_data_size = 0

      if size_hint:
        elements_data_size = size_hint.byte_size

      if not size_hint or not size_hint.is_complete:
        elements_data_size += self._element_data_type_definition.GetByteSize()

    return elements_data_size
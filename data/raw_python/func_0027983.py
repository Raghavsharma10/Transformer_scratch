def GetSizeHint(self, context=None, **unused_kwargs):
    """Retrieves a hint about the size.

    Args:
      context (Optional[DataTypeMapContext]): data type map context, used to
          determine the size hint.

    Returns:
      int: hint of the number of bytes needed from the byte stream or None.
    """
    context_state = getattr(context, 'state', {})

    subcontext = context_state.get('context', None)
    if not subcontext:
      mapped_values = context_state.get('mapped_values', None)
      subcontext = DataTypeMapContext(values={
          type(mapped_values).__name__: mapped_values})

    size_hint = 0
    for data_type_map in self._data_type_maps:
      data_type_size = data_type_map.GetSizeHint(context=subcontext)
      if data_type_size is None:
        break

      size_hint += data_type_size

    return size_hint
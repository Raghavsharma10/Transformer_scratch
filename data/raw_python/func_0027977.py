def _CompositeFoldByteStream(
      self, mapped_value, context=None, **unused_kwargs):
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
    context_state = getattr(context, 'state', {})

    attribute_index = context_state.get('attribute_index', 0)
    subcontext = context_state.get('context', None)

    if not subcontext:
      subcontext = DataTypeMapContext(values={
          type(mapped_value).__name__: mapped_value})

    data_attributes = []

    for attribute_index in range(attribute_index, self._number_of_attributes):
      attribute_name = self._attribute_names[attribute_index]
      data_type_map = self._data_type_maps[attribute_index]

      member_value = getattr(mapped_value, attribute_name, None)
      if data_type_map is None or member_value is None:
        continue

      member_data = data_type_map.FoldByteStream(
          member_value, context=subcontext)
      if member_data is None:
        return None

      data_attributes.append(member_data)

    if context:
      context.state = {}

    return b''.join(data_attributes)
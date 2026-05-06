def _CompositeMapByteStream(
      self, byte_stream, byte_offset=0, context=None, **unused_kwargs):
    """Maps a sequence of composite data types on a byte stream.

    Args:
      byte_stream (bytes): byte stream.
      byte_offset (Optional[int]): offset into the byte stream where to start.
      context (Optional[DataTypeMapContext]): data type map context.

    Returns:
      object: mapped value.

    Raises:
      MappingError: if the data type definition cannot be mapped on
          the byte stream.
    """
    context_state = getattr(context, 'state', {})

    attribute_index = context_state.get('attribute_index', 0)
    mapped_values = context_state.get('mapped_values', None)
    subcontext = context_state.get('context', None)

    if not mapped_values:
      mapped_values = self._structure_values_class()
    if not subcontext:
      subcontext = DataTypeMapContext(values={
          type(mapped_values).__name__: mapped_values})

    members_data_size = 0

    for attribute_index in range(attribute_index, self._number_of_attributes):
      attribute_name = self._attribute_names[attribute_index]
      data_type_map = self._data_type_maps[attribute_index]
      member_definition = self._data_type_definition.members[attribute_index]

      condition = getattr(member_definition, 'condition', None)
      if condition:
        namespace = dict(subcontext.values)
        # Make sure __builtins__ contains an empty dictionary.
        namespace['__builtins__'] = {}

        try:
          condition_result = eval(condition, namespace)  # pylint: disable=eval-used
        except Exception as exception:
          raise errors.MappingError(
              'Unable to evaluate condition with error: {0!s}'.format(
                  exception))

        if not isinstance(condition_result, bool):
          raise errors.MappingError(
              'Condition does not result in a boolean value')

        if not condition_result:
          continue

      if isinstance(member_definition, data_types.PaddingDefinition):
        _, byte_size = divmod(
            members_data_size, member_definition.alignment_size)
        if byte_size > 0:
          byte_size = member_definition.alignment_size - byte_size

        data_type_map.byte_size = byte_size

      try:
        value = data_type_map.MapByteStream(
            byte_stream, byte_offset=byte_offset, context=subcontext)
        setattr(mapped_values, attribute_name, value)

      except errors.ByteStreamTooSmallError as exception:
        context_state['attribute_index'] = attribute_index
        context_state['context'] = subcontext
        context_state['mapped_values'] = mapped_values
        raise errors.ByteStreamTooSmallError(exception)

      except Exception as exception:
        raise errors.MappingError(exception)

      supported_values = getattr(member_definition, 'values', None)
      if supported_values and value not in supported_values:
        raise errors.MappingError(
            'Value: {0!s} not in supported values: {1:s}'.format(
                value, ', '.join([
                    '{0!s}'.format(value) for value in supported_values])))

      byte_offset += subcontext.byte_size
      members_data_size += subcontext.byte_size

    if attribute_index != (self._number_of_attributes - 1):
      context_state['attribute_index'] = attribute_index
      context_state['context'] = subcontext
      context_state['mapped_values'] = mapped_values

      error_string = (
          'Unable to read: {0:s} from byte stream at offset: {1:d} '
          'with error: missing attribute: {2:d}').format(
              self._data_type_definition.name, byte_offset, attribute_index)
      raise errors.ByteStreamTooSmallError(error_string)

    if context:
      context.byte_size = members_data_size
      context.state = {}

    return mapped_values
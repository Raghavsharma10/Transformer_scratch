def _EvaluateElementsDataSize(self, context):
    """Evaluates elements data size.

    Args:
      context (DataTypeMapContext): data type map context.

    Returns:
      int: elements data size.

    Raises:
      MappingError: if the elements data size cannot be determined.
    """
    elements_data_size = None
    if self._data_type_definition.elements_data_size:
      elements_data_size = self._data_type_definition.elements_data_size

    elif self._data_type_definition.elements_data_size_expression:
      expression = self._data_type_definition.elements_data_size_expression
      namespace = {}
      if context and context.values:
        namespace.update(context.values)
      # Make sure __builtins__ contains an empty dictionary.
      namespace['__builtins__'] = {}

      try:
        elements_data_size = eval(expression, namespace)  # pylint: disable=eval-used
      except Exception as exception:
        raise errors.MappingError(
            'Unable to determine elements data size with error: {0!s}'.format(
                exception))

    if elements_data_size is None or elements_data_size < 0:
      raise errors.MappingError(
          'Invalid elements data size: {0!s}'.format(elements_data_size))

    return elements_data_size
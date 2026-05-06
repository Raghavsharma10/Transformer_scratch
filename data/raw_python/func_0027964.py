def _EvaluateNumberOfElements(self, context):
    """Evaluates number of elements.

    Args:
      context (DataTypeMapContext): data type map context.

    Returns:
      int: number of elements.

    Raises:
      MappingError: if the number of elements cannot be determined.
    """
    number_of_elements = None
    if self._data_type_definition.number_of_elements:
      number_of_elements = self._data_type_definition.number_of_elements

    elif self._data_type_definition.number_of_elements_expression:
      expression = self._data_type_definition.number_of_elements_expression
      namespace = {}
      if context and context.values:
        namespace.update(context.values)
      # Make sure __builtins__ contains an empty dictionary.
      namespace['__builtins__'] = {}

      try:
        number_of_elements = eval(expression, namespace)  # pylint: disable=eval-used
      except Exception as exception:
        raise errors.MappingError(
            'Unable to determine number of elements with error: {0!s}'.format(
                exception))

    if number_of_elements is None or number_of_elements < 0:
      raise errors.MappingError(
          'Invalid number of elements: {0!s}'.format(number_of_elements))

    return number_of_elements
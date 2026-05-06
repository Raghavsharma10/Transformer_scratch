def _CalculateElementsDataSize(self, context):
    """Calculates the elements data size.

    Args:
      context (Optional[DataTypeMapContext]): data type map context, used to
          determine the size hint.

    Returns:
      int: the elements data size or None if not available.
    """
    elements_data_size = None

    if self._HasElementsDataSize():
      elements_data_size = self._EvaluateElementsDataSize(context)

    elif self._HasNumberOfElements():
      element_byte_size = self._element_data_type_definition.GetByteSize()
      if element_byte_size is not None:
        number_of_elements = self._EvaluateNumberOfElements(context)
        elements_data_size = number_of_elements * element_byte_size

    return elements_data_size
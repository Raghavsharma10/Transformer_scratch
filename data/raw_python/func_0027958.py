def FoldValue(self, value):
    """Folds the data type into a value.

    Args:
      value (object): value.

    Returns:
      object: folded value.

    Raises:
      ValueError: if the data type definition cannot be folded into the value.
    """
    if value is False and self._data_type_definition.false_value is not None:
      return self._data_type_definition.false_value

    if value is True and self._data_type_definition.true_value is not None:
      return self._data_type_definition.true_value

    raise ValueError('No matching True and False values')
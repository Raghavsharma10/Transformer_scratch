def CreatePrecisionHelper(cls, precision):
    """Creates a precision helper.

    Args:
      precision (str): precision of the date and time value, which should
          be one of the PRECISION_VALUES in definitions.

    Returns:
      class: date time precision helper class.

    Raises:
      ValueError: if the precision value is unsupported.
    """
    precision_helper_class = cls._PRECISION_CLASSES.get(precision, None)
    if not precision_helper_class:
      raise ValueError('Unsupported precision: {0!s}'.format(precision))

    return precision_helper_class
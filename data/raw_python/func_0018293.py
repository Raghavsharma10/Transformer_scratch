def CopyMicrosecondsToFractionOfSecond(cls, microseconds):
    """Copies the number of microseconds to a fraction of second value.

    Args:
      microseconds (int): number of microseconds.

    Returns:
      decimal.Decimal: fraction of second, which must be a value between 0.0 and
          1.0.

    Raises:
      ValueError: if the number of microseconds is out of bounds.
    """
    if microseconds < 0 or microseconds >= definitions.MICROSECONDS_PER_SECOND:
      raise ValueError(
          'Number of microseconds value: {0:d} out of bounds.'.format(
              microseconds))

    return decimal.Decimal(microseconds) / definitions.MICROSECONDS_PER_SECOND
def CopyFromStringTuple(self, time_elements_tuple):
    """Copies time elements from string-based time elements tuple.

    Args:
      time_elements_tuple (Optional[tuple[str, str, str, str, str, str, str]]):
          time elements, contains year, month, day of month, hours, minutes,
          seconds and microseconds.

    Raises:
      ValueError: if the time elements tuple is invalid.
    """
    if len(time_elements_tuple) < 7:
      raise ValueError((
          'Invalid time elements tuple at least 7 elements required,'
          'got: {0:d}').format(len(time_elements_tuple)))

    year, month, day_of_month, hours, minutes, seconds, microseconds = (
        time_elements_tuple)

    try:
      microseconds = int(microseconds, 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid microsecond value: {0!s}'.format(microseconds))

    if microseconds < 0 or microseconds >= definitions.MICROSECONDS_PER_SECOND:
      raise ValueError('Invalid number of microseconds.')

    fraction_of_second = (
        decimal.Decimal(microseconds) / definitions.MICROSECONDS_PER_SECOND)

    time_elements_tuple = (
        year, month, day_of_month, hours, minutes, seconds,
        str(fraction_of_second))

    super(TimeElementsInMicroseconds, self).CopyFromStringTuple(
        time_elements_tuple)
def CopyFromStringTuple(self, time_elements_tuple):
    """Copies time elements from string-based time elements tuple.

    Args:
      time_elements_tuple (Optional[tuple[str, str, str, str, str, str, str]]):
          time elements, contains year, month, day of month, hours, minutes,
          seconds and milliseconds.

    Raises:
      ValueError: if the time elements tuple is invalid.
    """
    if len(time_elements_tuple) < 7:
      raise ValueError((
          'Invalid time elements tuple at least 7 elements required,'
          'got: {0:d}').format(len(time_elements_tuple)))

    year, month, day_of_month, hours, minutes, seconds, milliseconds = (
        time_elements_tuple)

    try:
      milliseconds = int(milliseconds, 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid millisecond value: {0!s}'.format(milliseconds))

    if milliseconds < 0 or milliseconds >= definitions.MILLISECONDS_PER_SECOND:
      raise ValueError('Invalid number of milliseconds.')

    fraction_of_second = (
        decimal.Decimal(milliseconds) / definitions.MILLISECONDS_PER_SECOND)

    time_elements_tuple = (
        year, month, day_of_month, hours, minutes, seconds,
        str(fraction_of_second))

    super(TimeElementsInMilliseconds, self).CopyFromStringTuple(
        time_elements_tuple)
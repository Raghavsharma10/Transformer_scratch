def CopyFromStringTuple(self, time_elements_tuple):
    """Copies time elements from string-based time elements tuple.

    Args:
      time_elements_tuple (Optional[tuple[str, str, str, str, str, str, str]]):
          time elements, contains year, month, day of month, hours, minutes,
          seconds and fraction of seconds.

    Raises:
      ValueError: if the time elements tuple is invalid.
    """
    if len(time_elements_tuple) < 7:
      raise ValueError((
          'Invalid time elements tuple at least 7 elements required,'
          'got: {0:d}').format(len(time_elements_tuple)))

    super(TimeElementsWithFractionOfSecond, self).CopyFromStringTuple(
        time_elements_tuple)

    try:
      fraction_of_second = decimal.Decimal(time_elements_tuple[6])
    except (TypeError, ValueError):
      raise ValueError('Invalid fraction of second value: {0!s}'.format(
          time_elements_tuple[6]))

    if fraction_of_second < 0.0 or fraction_of_second >= 1.0:
      raise ValueError('Fraction of second value: {0:f} out of bounds.'.format(
          fraction_of_second))

    self.fraction_of_second = fraction_of_second
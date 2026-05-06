def CopyToDateTimeString(cls, time_elements_tuple, fraction_of_second):
    """Copies the time elements and fraction of second to a string.

    Args:
      time_elements_tuple (tuple[int, int, int, int, int, int]):
          time elements, contains year, month, day of month, hours, minutes and
          seconds.
      fraction_of_second (decimal.Decimal): fraction of second, which must be a
          value between 0.0 and 1.0.

    Returns:
      str: date and time value formatted as:
          YYYY-MM-DD hh:mm:ss

    Raises:
      ValueError: if the fraction of second is out of bounds.
    """
    if fraction_of_second < 0.0 or fraction_of_second >= 1.0:
      raise ValueError('Fraction of second value: {0:f} out of bounds.'.format(
          fraction_of_second))

    return '{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'.format(
        time_elements_tuple[0], time_elements_tuple[1], time_elements_tuple[2],
        time_elements_tuple[3], time_elements_tuple[4], time_elements_tuple[5])
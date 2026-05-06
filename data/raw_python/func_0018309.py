def CopyFromStringTuple(self, time_elements_tuple):
    """Copies time elements from string-based time elements tuple.

    Args:
      time_elements_tuple (Optional[tuple[str, str, str, str, str, str]]):
          time elements, contains year, month, day of month, hours, minutes and
          seconds.

    Raises:
      ValueError: if the time elements tuple is invalid.
    """
    if len(time_elements_tuple) < 6:
      raise ValueError((
          'Invalid time elements tuple at least 6 elements required,'
          'got: {0:d}').format(len(time_elements_tuple)))

    try:
      year = int(time_elements_tuple[0], 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid year value: {0!s}'.format(
          time_elements_tuple[0]))

    try:
      month = int(time_elements_tuple[1], 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid month value: {0!s}'.format(
          time_elements_tuple[1]))

    try:
      day_of_month = int(time_elements_tuple[2], 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid day of month value: {0!s}'.format(
          time_elements_tuple[2]))

    try:
      hours = int(time_elements_tuple[3], 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid hours value: {0!s}'.format(
          time_elements_tuple[3]))

    try:
      minutes = int(time_elements_tuple[4], 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid minutes value: {0!s}'.format(
          time_elements_tuple[4]))

    try:
      seconds = int(time_elements_tuple[5], 10)
    except (TypeError, ValueError):
      raise ValueError('Invalid seconds value: {0!s}'.format(
          time_elements_tuple[5]))

    self._normalized_timestamp = None
    self._number_of_seconds = self._GetNumberOfSecondsFromElements(
        year, month, day_of_month, hours, minutes, seconds)
    self._time_elements_tuple = (
        year, month, day_of_month, hours, minutes, seconds)
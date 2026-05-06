def CopyToDateTimeString(self):
    """Copies the time elements to a date and time string.

    Returns:
      str: date and time value formatted as: "YYYY-MM-DD hh:mm:ss" or None
          if time elements are missing.
    """
    if self._number_of_seconds is None:
      return None

    return '{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}'.format(
        self._time_elements_tuple[0], self._time_elements_tuple[1],
        self._time_elements_tuple[2], self._time_elements_tuple[3],
        self._time_elements_tuple[4], self._time_elements_tuple[5])
def CopyToDateTimeString(self):
    """Copies the RFC2579 date-time to a date and time string.

    Returns:
      str: date and time value formatted as: "YYYY-MM-DD hh:mm:ss.#" or
          None if the number of seconds is missing.
    """
    if self._number_of_seconds is None:
      return None

    return '{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}.{6:01d}'.format(
        self.year, self.month, self.day_of_month, self.hours, self.minutes,
        self.seconds, self.deciseconds)
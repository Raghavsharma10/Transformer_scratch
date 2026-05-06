def CopyFromDateTimeString(self, time_string):
    """Copies a Delphi TDateTime timestamp from a string.

    Args:
      time_string (str): date and time value formatted as:
          YYYY-MM-DD hh:mm:ss.######[+-]##:##

          Where # are numeric digits ranging from 0 to 9 and the seconds
          fraction can be either 3 or 6 digits. The time of day, seconds
          fraction and time zone offset are optional. The default time zone
          is UTC.

    Raises:
      ValueError: if the time string is invalid or not supported.
    """
    date_time_values = self._CopyDateTimeFromString(time_string)

    year = date_time_values.get('year', 0)
    month = date_time_values.get('month', 0)
    day_of_month = date_time_values.get('day_of_month', 0)
    hours = date_time_values.get('hours', 0)
    minutes = date_time_values.get('minutes', 0)
    seconds = date_time_values.get('seconds', 0)
    microseconds = date_time_values.get('microseconds', None)

    if year > 9999:
      raise ValueError('Unsupported year value: {0:d}.'.format(year))

    timestamp = self._GetNumberOfSecondsFromElements(
        year, month, day_of_month, hours, minutes, seconds)

    timestamp = float(timestamp) / definitions.SECONDS_PER_DAY
    timestamp += self._DELPHI_TO_POSIX_BASE
    if microseconds is not None:
      timestamp += float(microseconds) / definitions.MICROSECONDS_PER_DAY

    self._normalized_timestamp = None
    self._timestamp = timestamp
    self.is_local_time = False
def CopyFromDateTimeString(self, time_string):
    """Copies a APFS timestamp from a date and time string.

    Args:
      time_string (str): date and time value formatted as:
          YYYY-MM-DD hh:mm:ss.######[+-]##:##

          Where # are numeric digits ranging from 0 to 9 and the seconds
          fraction can be either 3 or 6 digits. The time of day, seconds
          fraction and time zone offset are optional. The default time zone
          is UTC.

    Raises:
      ValueError: if the date and time value is not supported.
    """
    super(APFSTime, self)._CopyFromDateTimeString(time_string)

    if (self._timestamp is None or self._timestamp < self._INT64_MIN or
        self._timestamp > self._INT64_MAX):
      raise ValueError('Date time value not supported.')
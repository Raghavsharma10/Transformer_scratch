def _CopyDateTimeFromStringISO8601(self, time_string):
    """Copies a date and time from an ISO 8601 date and time string.

    Args:
      time_string (str): time value formatted as:
          hh:mm:ss.######[+-]##:##

          Where # are numeric digits ranging from 0 to 9 and the seconds
          fraction can be either 3 or 6 digits. The fraction of second and
          time zone offset are optional.

    Returns:
      tuple[int, int, int, int, int]: hours, minutes, seconds, microseconds,
          time zone offset in minutes.

    Raises:
      ValueError: if the time string is invalid or not supported.
    """
    if not time_string:
      raise ValueError('Invalid time string.')

    time_string_length = len(time_string)

    year, month, day_of_month = self._CopyDateFromString(time_string)

    if time_string_length <= 10:
      return {
          'year': year,
          'month': month,
          'day_of_month': day_of_month}

    # If a time of day is specified the time string it should at least
    # contain 'YYYY-MM-DDThh'.
    if time_string[10] != 'T':
      raise ValueError(
          'Invalid time string - missing as date and time separator.')

    hours, minutes, seconds, microseconds, time_zone_offset = (
        self._CopyTimeFromStringISO8601(time_string[11:]))

    if time_zone_offset:
      year, month, day_of_month, hours, minutes = self._AdjustForTimeZoneOffset(
          year, month, day_of_month, hours, minutes, time_zone_offset)

    date_time_values = {
        'year': year,
        'month': month,
        'day_of_month': day_of_month,
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds}

    if microseconds is not None:
      date_time_values['microseconds'] = microseconds
    return date_time_values
def _CopyDateFromString(self, date_string):
    """Copies a date from a string.

    Args:
      date_string (str): date value formatted as: YYYY-MM-DD

    Returns:
      tuple[int, int, int]: year, month, day of month.

    Raises:
      ValueError: if the date string is invalid or not supported.
    """
    date_string_length = len(date_string)

    # The date string should at least contain 'YYYY-MM-DD'.
    if date_string_length < 10:
      raise ValueError('Date string too short.')

    if date_string[4] != '-' or date_string[7] != '-':
      raise ValueError('Invalid date string.')

    try:
      year = int(date_string[0:4], 10)
    except ValueError:
      raise ValueError('Unable to parse year.')

    try:
      month = int(date_string[5:7], 10)
    except ValueError:
      raise ValueError('Unable to parse month.')

    try:
      day_of_month = int(date_string[8:10], 10)
    except ValueError:
      raise ValueError('Unable to parse day of month.')

    days_per_month = self._GetDaysPerMonth(year, month)
    if day_of_month < 1 or day_of_month > days_per_month:
      raise ValueError('Day of month value out of bounds.')

    return year, month, day_of_month
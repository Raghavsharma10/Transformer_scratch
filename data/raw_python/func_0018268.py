def _GetNumberOfSecondsFromElements(
      self, year, month, day_of_month, hours, minutes, seconds):
    """Retrieves the number of seconds from the date and time elements.

    Args:
      year (int): year e.g. 1970.
      month (int): month, where 1 represents January.
      day_of_month (int): day of the month, where 1 represents the first day.
      hours (int): hours.
      minutes (int): minutes.
      seconds (int): seconds.

    Returns:
      int: number of seconds since January 1, 1970 00:00:00 or None if year,
          month or day of month are not set.

    Raises:
      ValueError: if the time elements are invalid.
    """
    if not year or not month or not day_of_month:
      return None

    # calendar.timegm does not sanity check the time elements.
    if hours is None:
      hours = 0
    elif hours not in range(0, 24):
      raise ValueError('Hours value: {0!s} out of bounds.'.format(hours))

    if minutes is None:
      minutes = 0
    elif minutes not in range(0, 60):
      raise ValueError('Minutes value: {0!s} out of bounds.'.format(minutes))

    # TODO: support a leap second?
    if seconds is None:
      seconds = 0
    elif seconds not in range(0, 60):
      raise ValueError('Seconds value: {0!s} out of bounds.'.format(seconds))

    # Note that calendar.timegm() does not raise when date is: 2013-02-29.
    days_per_month = self._GetDaysPerMonth(year, month)
    if day_of_month < 1 or day_of_month > days_per_month:
      raise ValueError('Day of month value out of bounds.')

    # calendar.timegm requires the time tuple to contain at least
    # 6 integer values.
    time_elements_tuple = (year, month, day_of_month, hours, minutes, seconds)

    number_of_seconds = calendar.timegm(time_elements_tuple)

    return int(number_of_seconds)
def _GetNumberOfSeconds(self, fat_date_time):
    """Retrieves the number of seconds from a FAT date time.

    Args:
      fat_date_time (int): FAT date time.

    Returns:
      int: number of seconds since January 1, 1980 00:00:00.

    Raises:
      ValueError: if the month, day of month, hours, minutes or seconds
          value is out of bounds.
    """
    day_of_month = (fat_date_time & 0x1f)
    month = ((fat_date_time >> 5) & 0x0f)
    year = (fat_date_time >> 9) & 0x7f

    days_per_month = self._GetDaysPerMonth(year, month)
    if day_of_month < 1 or day_of_month > days_per_month:
      raise ValueError('Day of month value out of bounds.')

    number_of_days = self._GetDayOfYear(1980 + year, month, day_of_month)
    number_of_days -= 1
    for past_year in range(0, year):
      number_of_days += self._GetNumberOfDaysInYear(past_year)

    fat_date_time >>= 16

    seconds = (fat_date_time & 0x1f) * 2
    minutes = (fat_date_time >> 5) & 0x3f
    hours = (fat_date_time >> 11) & 0x1f

    if hours not in range(0, 24):
      raise ValueError('Hours value out of bounds.')

    if minutes not in range(0, 60):
      raise ValueError('Minutes value out of bounds.')

    if seconds not in range(0, 60):
      raise ValueError('Seconds value out of bounds.')

    number_of_seconds = (((hours * 60) + minutes) * 60) + seconds
    number_of_seconds += number_of_days * definitions.SECONDS_PER_DAY
    return number_of_seconds
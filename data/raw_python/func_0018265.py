def _GetDayOfYear(self, year, month, day_of_month):
    """Retrieves the day of the year for a specific day of a month in a year.

    Args:
      year (int): year e.g. 1970.
      month (int): month, where 1 represents January.
      day_of_month (int): day of the month, where 1 represents the first day.

    Returns:
      int: day of year.

    Raises:
      ValueError: if the month or day of month value is out of bounds.
    """
    if month not in range(1, 13):
      raise ValueError('Month value out of bounds.')

    days_per_month = self._GetDaysPerMonth(year, month)
    if day_of_month < 1 or day_of_month > days_per_month:
      raise ValueError('Day of month value out of bounds.')

    day_of_year = day_of_month
    for past_month in range(1, month):
      day_of_year += self._GetDaysPerMonth(year, past_month)

    return day_of_year
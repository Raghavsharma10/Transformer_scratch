def _GetDaysPerMonth(self, year, month):
    """Retrieves the number of days in a month of a specific year.

    Args:
      year (int): year e.g. 1970.
      month (int): month, where 1 represents January.

    Returns:
      int: number of days in the month.

    Raises:
      ValueError: if the month value is out of bounds.
    """
    if month not in range(1, 13):
      raise ValueError('Month value out of bounds.')

    days_per_month = self._DAYS_PER_MONTH[month - 1]
    if month == 2 and self._IsLeapYear(year):
      days_per_month += 1

    return days_per_month
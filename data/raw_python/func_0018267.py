def _GetNumberOfDaysInCentury(self, year):
    """Retrieves the number of days in a century.

    Args:
      year (int): year in the century e.g. 1970.

    Returns:
      int: number of (remaining) days in the century.

    Raises:
      ValueError: if the year value is out of bounds.
    """
    if year < 0:
      raise ValueError('Year value out of bounds.')

    year, _ = divmod(year, 100)

    if self._IsLeapYear(year):
      return 36525
    return 36524
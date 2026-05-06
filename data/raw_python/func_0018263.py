def _GetDateValues(
      self, number_of_days, epoch_year, epoch_month, epoch_day_of_month):
    """Determines date values.

    Args:
      number_of_days (int): number of days since epoch.
      epoch_year (int): year that is the start of the epoch e.g. 1970.
      epoch_month (int): month that is the start of the epoch, where
          1 represents January.
      epoch_day_of_month (int): day of month that is the start of the epoch,
          where 1 represents the first day.

    Returns:
       tuple[int, int, int]: year, month, day of month.

    Raises:
      ValueError: if the epoch year, month or day of month values are out
          of bounds.
    """
    if epoch_year < 0:
      raise ValueError('Epoch year value: {0:d} out of bounds.'.format(
          epoch_year))

    if epoch_month not in range(1, 13):
      raise ValueError('Epoch month value: {0:d} out of bounds.'.format(
          epoch_month))

    epoch_days_per_month = self._GetDaysPerMonth(epoch_year, epoch_month)
    if epoch_day_of_month < 1 or epoch_day_of_month > epoch_days_per_month:
      raise ValueError('Epoch day of month value: {0:d} out of bounds.'.format(
          epoch_day_of_month))

    before_epoch = number_of_days < 0

    year = epoch_year
    month = epoch_month
    if before_epoch:
      month -= 1
      if month <= 0:
        month = 12
        year -= 1

    number_of_days += epoch_day_of_month
    if before_epoch:
      number_of_days *= -1

    # Align with the start of the year.
    while month > 1:
      days_per_month = self._GetDaysPerMonth(year, month)
      if number_of_days < days_per_month:
        break

      if before_epoch:
        month -= 1
      else:
        month += 1

      if month > 12:
        month = 1
        year += 1

      number_of_days -= days_per_month

    # Align with the start of the next century.
    _, remainder = divmod(year, 100)
    for _ in range(remainder, 100):
      days_in_year = self._GetNumberOfDaysInYear(year)
      if number_of_days < days_in_year:
        break

      if before_epoch:
        year -= 1
      else:
        year += 1

      number_of_days -= days_in_year

    days_in_century = self._GetNumberOfDaysInCentury(year)
    while number_of_days > days_in_century:
      if before_epoch:
        year -= 100
      else:
        year += 100

      number_of_days -= days_in_century
      days_in_century = self._GetNumberOfDaysInCentury(year)

    days_in_year = self._GetNumberOfDaysInYear(year)
    while number_of_days > days_in_year:
      if before_epoch:
        year -= 1
      else:
        year += 1

      number_of_days -= days_in_year
      days_in_year = self._GetNumberOfDaysInYear(year)

    days_per_month = self._GetDaysPerMonth(year, month)
    while number_of_days > days_per_month:
      if before_epoch:
        month -= 1
      else:
        month += 1

      if month <= 0:
        month = 12
        year -= 1
      elif month > 12:
        month = 1
        year += 1

      number_of_days -= days_per_month
      days_per_month = self._GetDaysPerMonth(year, month)

    if before_epoch:
      days_per_month = self._GetDaysPerMonth(year, month)
      number_of_days = days_per_month - number_of_days

    elif number_of_days == 0:
      number_of_days = 31
      month = 12
      year -= 1

    return year, month, number_of_days
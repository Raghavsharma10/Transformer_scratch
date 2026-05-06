def _GetDateValuesWithEpoch(self, number_of_days, date_time_epoch):
    """Determines date values.

    Args:
      number_of_days (int): number of days since epoch.
      date_time_epoch (DateTimeEpoch): date and time of the epoch.

    Returns:
       tuple[int, int, int]: year, month, day of month.
    """
    return self._GetDateValues(
        number_of_days, date_time_epoch.year, date_time_epoch.month,
        date_time_epoch.day_of_month)
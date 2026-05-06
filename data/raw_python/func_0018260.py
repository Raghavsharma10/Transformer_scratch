def _AdjustForTimeZoneOffset(
      self, year, month, day_of_month, hours, minutes, time_zone_offset):
    """Adjusts the date and time values for a time zone offset.

    Args:
      year (int): year e.g. 1970.
      month (int): month, where 1 represents January.
      day_of_month (int): day of the month, where 1 represents the first day.
      hours (int): hours.
      minutes (int): minutes.
      time_zone_offset (int): time zone offset in number of minutes from UTC.

    Returns:
      tuple[int, int, int, int, int, int]: time zone correct year, month,
         day_of_month, hours and minutes values.
    """
    hours_from_utc, minutes_from_utc = divmod(time_zone_offset, 60)

    minutes += minutes_from_utc

    # Since divmod makes sure the sign of minutes_from_utc is positive
    # we only need to check the upper bound here, because hours_from_utc
    # remains signed it is corrected accordingly.
    if minutes >= 60:
      minutes -= 60
      hours += 1

    hours += hours_from_utc
    if hours < 0:
      hours += 24
      day_of_month -= 1

    elif hours >= 24:
      hours -= 24
      day_of_month += 1

    days_per_month = self._GetDaysPerMonth(year, month)
    if day_of_month < 1:
      month -= 1
      if month < 1:
        month = 12
        year -= 1

      day_of_month += self._GetDaysPerMonth(year, month)

    elif day_of_month > days_per_month:
      month += 1
      if month > 12:
        month = 1
        year += 1

      day_of_month -= days_per_month

    return year, month, day_of_month, hours, minutes
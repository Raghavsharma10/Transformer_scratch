def CopyToDateTimeString(self):
    """Copies the FILETIME timestamp to a date and time string.

    Returns:
      str: date and time value formatted as: "YYYY-MM-DD hh:mm:ss.#######" or
          None if the timestamp is missing or invalid.
    """
    if (self._timestamp is None or self._timestamp < 0 or
        self._timestamp > self._UINT64_MAX):
      return None

    timestamp, remainder = divmod(self._timestamp, self._100NS_PER_SECOND)
    number_of_days, hours, minutes, seconds = self._GetTimeValues(timestamp)

    year, month, day_of_month = self._GetDateValuesWithEpoch(
        number_of_days, self._EPOCH)

    return '{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}.{6:07d}'.format(
        year, month, day_of_month, hours, minutes, seconds, remainder)
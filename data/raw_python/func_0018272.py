def GetDate(self):
    """Retrieves the date represented by the date and time values.

    Returns:
       tuple[int, int, int]: year, month, day of month or (None, None, None)
           if the date and time values do not represent a date.
    """
    normalized_timestamp = self._GetNormalizedTimestamp()
    if normalized_timestamp is None:
      return None, None, None

    number_of_days, _, _, _ = self._GetTimeValues(normalized_timestamp)

    try:
      return self._GetDateValuesWithEpoch(
          number_of_days, self._EPOCH_NORMALIZED_TIME)

    except ValueError:
      return None, None, None
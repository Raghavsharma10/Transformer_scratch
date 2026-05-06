def GetTimeOfDay(self):
    """Retrieves the time of day represented by the date and time values.

    Returns:
       tuple[int, int, int]: hours, minutes, seconds or (None, None, None)
           if the date and time values do not represent a time of day.
    """
    normalized_timestamp = self._GetNormalizedTimestamp()
    if normalized_timestamp is None:
      return None, None, None

    _, hours, minutes, seconds = self._GetTimeValues(normalized_timestamp)
    return hours, minutes, seconds
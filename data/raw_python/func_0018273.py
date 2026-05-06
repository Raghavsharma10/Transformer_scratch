def GetPlasoTimestamp(self):
    """Retrieves a timestamp that is compatible with plaso.

    Returns:
      int: a POSIX timestamp in microseconds or None if no timestamp is
          available.
    """
    normalized_timestamp = self._GetNormalizedTimestamp()
    if normalized_timestamp is None:
      return None

    normalized_timestamp *= definitions.MICROSECONDS_PER_SECOND
    normalized_timestamp = normalized_timestamp.quantize(
        1, rounding=decimal.ROUND_HALF_UP)
    return int(normalized_timestamp)
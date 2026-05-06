def CopyToStatTimeTuple(self):
    """Copies the date time value to a stat timestamp tuple.

    Returns:
      tuple[int, int]: a POSIX timestamp in seconds and the remainder in
          100 nano seconds or (None, None) on error.
    """
    normalized_timestamp = self._GetNormalizedTimestamp()
    if normalized_timestamp is None:
      return None, None

    if self._precision in (
        definitions.PRECISION_1_NANOSECOND,
        definitions.PRECISION_100_NANOSECONDS,
        definitions.PRECISION_1_MICROSECOND,
        definitions.PRECISION_1_MILLISECOND,
        definitions.PRECISION_100_MILLISECONDS):
      remainder = int((normalized_timestamp % 1) * self._100NS_PER_SECOND)

      return int(normalized_timestamp), remainder

    return int(normalized_timestamp), None
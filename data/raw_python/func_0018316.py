def _GetNormalizedTimestamp(self):
    """Retrieves the normalized timestamp.

    Returns:
      decimal.Decimal: normalized timestamp, which contains the number of
          seconds since January 1, 1970 00:00:00 and a fraction of second used
          for increased precision, or None if the normalized timestamp cannot be
          determined.
    """
    if self._normalized_timestamp is None:
      if self._number_of_seconds is not None:
        self._normalized_timestamp = (
            decimal.Decimal(self.milliseconds) /
            definitions.MILLISECONDS_PER_SECOND)
        self._normalized_timestamp += decimal.Decimal(self._number_of_seconds)

    return self._normalized_timestamp
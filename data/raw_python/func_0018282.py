def _GetNormalizedTimestamp(self):
    """Retrieves the normalized timestamp.

    Returns:
      decimal.Decimal: normalized timestamp, which contains the number of
          seconds since January 1, 1970 00:00:00 and a fraction of second used
          for increased precision, or None if the normalized timestamp cannot be
          determined.
    """
    if self._normalized_timestamp is None:
      if self._number_of_seconds is not None and self._number_of_seconds >= 0:
        self._normalized_timestamp = (
            decimal.Decimal(self._number_of_seconds) +
            self._FAT_DATE_TO_POSIX_BASE)

    return self._normalized_timestamp
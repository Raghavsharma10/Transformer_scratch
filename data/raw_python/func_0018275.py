def _GetNormalizedTimestamp(self):
    """Retrieves the normalized timestamp.

    Returns:
      decimal.Decimal: normalized timestamp, which contains the number of
          seconds since January 1, 1970 00:00:00 and a fraction of second used
          for increased precision, or None if the normalized timestamp cannot be
          determined.
    """
    if self._normalized_timestamp is None:
      if (self._timestamp is not None and self._timestamp >= 0 and
          self._timestamp <= self._UINT32_MAX):
        self._normalized_timestamp = (
            decimal.Decimal(self._timestamp) - self._HFS_TO_POSIX_BASE)

    return self._normalized_timestamp
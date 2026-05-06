def CopyToDateTimeString(self):
    """Copies the APFS timestamp to a date and time string.

    Returns:
      str: date and time value formatted as: "YYYY-MM-DD hh:mm:ss.#########" or
          None if the timestamp is missing or invalid.
    """
    if (self._timestamp is None or self._timestamp < self._INT64_MIN or
        self._timestamp > self._INT64_MAX):
      return None

    return super(APFSTime, self)._CopyToDateTimeString()
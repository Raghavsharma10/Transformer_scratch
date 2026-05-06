def CopyToDateTimeString(self):
    """Copies the time elements to a date and time string.

    Returns:
      str: date and time value formatted as: "YYYY-MM-DD hh:mm:ss" or
          "YYYY-MM-DD hh:mm:ss.######" or None if time elements are missing.

    Raises:
      ValueError: if the precision value is unsupported.
    """
    if self._number_of_seconds is None or self.fraction_of_second is None:
      return None

    precision_helper = precisions.PrecisionHelperFactory.CreatePrecisionHelper(
        self._precision)

    return precision_helper.CopyToDateTimeString(
        self._time_elements_tuple, self.fraction_of_second)
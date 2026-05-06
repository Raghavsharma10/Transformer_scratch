def _CopyFromDateTimeValues(self, date_time_values):
    """Copies time elements from date and time values.

    Args:
      date_time_values  (dict[str, int]): date and time values, such as year,
          month, day of month, hours, minutes, seconds, microseconds.

    Raises:
      ValueError: if no helper can be created for the current precision.
    """
    year = date_time_values.get('year', 0)
    month = date_time_values.get('month', 0)
    day_of_month = date_time_values.get('day_of_month', 0)
    hours = date_time_values.get('hours', 0)
    minutes = date_time_values.get('minutes', 0)
    seconds = date_time_values.get('seconds', 0)
    microseconds = date_time_values.get('microseconds', 0)

    precision_helper = precisions.PrecisionHelperFactory.CreatePrecisionHelper(
        self._precision)

    fraction_of_second = precision_helper.CopyMicrosecondsToFractionOfSecond(
        microseconds)

    self._normalized_timestamp = None
    self._number_of_seconds = self._GetNumberOfSecondsFromElements(
        year, month, day_of_month, hours, minutes, seconds)
    self._time_elements_tuple = (
        year, month, day_of_month, hours, minutes, seconds)
    self.fraction_of_second = fraction_of_second
    self.is_local_time = False
def _CopyFromDateTimeValues(self, date_time_values):
    """Copies time elements from date and time values.

    Args:
      date_time_values  (dict[str, int]): date and time values, such as year,
          month, day of month, hours, minutes, seconds, microseconds.
    """
    year = date_time_values.get('year', 0)
    month = date_time_values.get('month', 0)
    day_of_month = date_time_values.get('day_of_month', 0)
    hours = date_time_values.get('hours', 0)
    minutes = date_time_values.get('minutes', 0)
    seconds = date_time_values.get('seconds', 0)

    self._normalized_timestamp = None
    self._number_of_seconds = self._GetNumberOfSecondsFromElements(
        year, month, day_of_month, hours, minutes, seconds)
    self._time_elements_tuple = (
        year, month, day_of_month, hours, minutes, seconds)

    self.is_local_time = False